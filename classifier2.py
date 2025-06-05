import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
import math
from sam import SAM

# Seed 설정 (재현 가능한 결과를 위해)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CutMix 구현
def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    
    return data, targets, shuffled_targets, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Mixup 함수
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Label Smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Wide ResNet 구현
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(BasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(BasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(BasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# EfficientNet-style 모델 (더 가벼운 버전)
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False) if expand_ratio != 1 else None
        self.expand_bn = nn.BatchNorm2d(expanded_channels) if expand_ratio != 1 else None
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride, 
                                      kernel_size//2, groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # SE block
        se_channels = max(1, int(in_channels * se_ratio))
        self.se_reduce = nn.Conv2d(expanded_channels, se_channels, 1, bias=True)
        self.se_expand = nn.Conv2d(se_channels, expanded_channels, 1, bias=True)
        
        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        
        self.swish = nn.SiLU()

    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = self.swish(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise
        x = self.swish(self.depthwise_bn(self.depthwise_conv(x)))
        
        # SE block
        se_weight = F.adaptive_avg_pool2d(x, 1)
        se_weight = self.se_expand(self.swish(self.se_reduce(se_weight)))
        se_weight = torch.sigmoid(se_weight)
        x = x * se_weight
        
        # Pointwise
        x = self.pointwise_bn(self.pointwise_conv(x))
        
        # Residual connection
        if self.use_residual:
            x = x + identity
            
        return x

class EfficientNetCIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNetCIFAR, self).__init__()
        
        # First conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # Stage 1
            MBConvBlock(32, 16, 3, 1, 1),
            # Stage 2
            MBConvBlock(16, 24, 3, 2, 6),
            MBConvBlock(24, 24, 3, 1, 6),
            # Stage 3
            MBConvBlock(24, 40, 5, 2, 6),
            MBConvBlock(40, 40, 5, 1, 6),
            # Stage 4
            MBConvBlock(40, 80, 3, 2, 6),
            MBConvBlock(80, 80, 3, 1, 6),
            MBConvBlock(80, 80, 3, 1, 6),
            # Stage 5
            MBConvBlock(80, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),
            # Stage 6
            MBConvBlock(112, 192, 5, 2, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            # Stage 7
            MBConvBlock(192, 320, 3, 1, 6),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# AutoAugment for CIFAR-100
class AutoAugmentTransform:
    def __init__(self):
        self.transforms = [
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        ]

    def __call__(self, img):
        return random.choice(self.transforms)(img)

# 강화된 데이터 로더
def get_cifar100_train_loader(batch_size=128, num_workers=4):
    print("Creating enhanced CIFAR-100 dataset with advanced augmentations...")
    
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    # 더 강력한 augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        AutoAugmentTransform(),  # AutoAugment 추가
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Random Erasing
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        drop_last=True  # 마지막 배치가 작을 때 문제 방지
    )
    
    return train_loader

def get_cifar100_val_loader(batch_size=128, num_workers=4):
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    valset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_val
    )

    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return val_loader

# 향상된 추론용 데이터 로딩 (TTA 적용)
def load_test_images_with_tta(test_dir, num_tta=5):
    image_paths = sorted([os.path.join(test_dir, fname)
                          for fname in os.listdir(test_dir)
                          if fname.endswith(('.png', '.jpg', '.jpeg'))])

    # TTA용 여러 변환
    base_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ]) for _ in range(num_tta)
    ]

    all_images = []
    filenames = []
    
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        
        # 원본 + TTA 이미지들
        images_for_this_file = [base_transform(img)]
        for transform in tta_transforms:
            images_for_this_file.append(transform(img))
        
        all_images.append(torch.stack(images_for_this_file))
        filenames.append(os.path.basename(path))

    return all_images, filenames

# 앙상블 모델 클래스
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(F.softmax(model(x), dim=1))
        return torch.mean(torch.stack(outputs), dim=0)

# 향상된 학습 루프
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=100):
    best_acc = 0.0
    patience = 20
    no_improve = 0
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # First forward-backward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass
            outputs = model(inputs)
            criterion(outputs, targets).backward()
            optimizer.second_step(zero_grad=True)
            
            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            
            # 정확도 계산 (근사치)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()

        # Validation
        val_acc = validate(model, val_loader, device)
        
        epoch_loss = running_loss / total
        train_acc = 100. * correct / total

        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # 모델 저장 및 조기 종료
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            model_path = "weight_가반1조_0602_1410.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model! Val Acc: {best_acc:.2f}%")
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
            
        # 스케줄러 업데이트 (epoch 기반)
        if hasattr(scheduler, 'step') and not hasattr(scheduler, 'step_batch'):
            scheduler.step()

    return best_acc

def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
    
    return 100. * correct / total

# TTA를 적용한 추론
def inference_with_tta(model_path, test_dir, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        print(f"Error: Weight file '{model_path}' not found!")
        return

    # 최고 성능 모델 로드 (앙상블용으로 여러 모델 사용 가능)
    model1 = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100).to(device)
    model2 = EfficientNetCIFAR(num_classes=100).to(device)
    
    try:
        # 첫 번째 모델만 로드 (실제로는 여러 모델을 따로 학습하여 앙상블)
        model1.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model1.eval()

    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found!")
        return

    all_images, filenames = load_test_images_with_tta(test_dir, num_tta=8)
    
    predictions = []
    for images_for_file in tqdm(all_images, desc="Inferencing with TTA"):
        images_for_file = images_for_file.to(device)
        
        with torch.no_grad():
            # TTA: 여러 augmentation에 대한 예측을 평균
            tta_outputs = []
            for img in images_for_file:
                img = img.unsqueeze(0)
                output = model1(img)
                tta_outputs.append(F.softmax(output, dim=1))
            
            # TTA 결과 평균
            final_output = torch.mean(torch.stack(tta_outputs), dim=0)
            pred = final_output.argmax(dim=1).cpu().item()
            predictions.append(pred)

    # 결과 저장
    with open(output_file, 'w') as f:
        f.write("number, label\n")
        for fname, pred in zip(filenames, predictions):
            number = os.path.splitext(fname)[0]
            formatted_number = f"{int(number):04d}"
            f.write(f"{formatted_number}, {pred:02d}\n")

    print(f"Saved inference results to {output_file}")

def main():
    set_seed(42)  # 재현 가능한 결과를 위해
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CUDA 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # ============== 학습 모드 ==============
    # 최고 성능을 위해 Wide ResNet 사용
    model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100).to(device)
    print("Wide ResNet-28-10 model created successfully")

    # 데이터 로더
    train_loader = get_cifar100_train_loader(batch_size=128, num_workers=4)
    val_loader = get_cifar100_val_loader(batch_size=128, num_workers=4)
    print("Data loaders created successfully")

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # 1. 먼저 base optimizer를 생성 (여기에 lr 설정)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

    # 2. SAM으로 감쌈 (lr 파라미터 없이)
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # 학습 실행
    best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=100)
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")

    # ============== 추론 모드 ==============
    # test_dir = "./Dataset/CImages"
    # model_path = "weight_가반1조_0602_1410.pth"
    # output_file = "result_가반1조_0602_1410.txt"
    # inference_with_tta(model_path, test_dir, output_file)

if __name__ == '__main__':
    main()
