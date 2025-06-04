import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from PIL import Image

# Mixup 함수 추가
def mixup_data(x, y, alpha=0.2):
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

# ------------------- 모델 정의 -------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ------------------- 데이터 전처리 및 로딩 -------------------
def get_cifar100_train_loader(batch_size=128, num_workers=2):
    print("Starting CIFAR-100 dataset download and processing...")
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    print("Downloading CIFAR-100 dataset...")
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    print("Dataset downloaded successfully!")

    print("Creating data loader...")
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    print("Data loader created successfully!")
    return train_loader

# ------------------- 추론용 데이터 로딩 -------------------
def load_test_images(test_dir):
    image_paths = sorted([os.path.join(test_dir, fname)
                          for fname in os.listdir(test_dir)
                          if fname.endswith(('.png', '.jpg', '.jpeg'))])

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    images = []
    filenames = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)
        filenames.append(os.path.basename(path))

    batch = torch.stack(images)
    return batch, filenames

# ------------------- 학습 루프 -------------------
def train(model, train_loader, criterion, optimizer, scheduler, device, epochs=50):
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixup 적용
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (lam * preds.eq(targets_a).sum().float() 
                      + (1 - lam) * preds.eq(targets_b).sum().float())
            total += targets.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            model_path = "weight_가반1조_0602_1410.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path} (acc: {best_acc:.2f}%)")

# ------------------- 추론 및 결과 저장 -------------------
def inference_and_save(model_path, test_dir, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes=100).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    inputs, filenames = load_test_images(test_dir)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()

    with open(output_file, 'w') as f:
        f.write("number, label\n")
        for fname, pred in zip(filenames, preds):
            number = os.path.splitext(fname)[0]
            # 4자리 숫자로 포맷팅
            formatted_number = f"{int(number):04d}"
            f.write(f"{formatted_number}, {pred:02d}\n")

    print(f"Saved inference results to {output_file}")

# ------------------- 메인 실행 -------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CUDA 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # ============== 학습 모드 (주석 해제하여 사용) ==============
    # model = SimpleCNN(num_classes=100).to(device)
    # print("Model created successfully")

    # train_loader = get_cifar100_train_loader(batch_size=64, num_workers=4)
    # print("Data loader created successfully")

    # criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # # OneCycleLR 스케줄러로 변경
    # steps_per_epoch = len(train_loader)
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=0.01,
    #     epochs=150,
    #     steps_per_epoch=steps_per_epoch,
    #     pct_start=0.3,
    #     anneal_strategy='cos'
    # )

    # train(model, train_loader, criterion, optimizer, scheduler, device, epochs=150)

    # ============== 추론 모드 (기본 활성화) ==============
    test_dir = "./Dataset/CImages"
    model_path = "weight_가반1조_0602_1410.pth"
    output_file = "result_가반1조_0602_1410.txt"
    inference_and_save(model_path, test_dir, output_file)

if __name__ == '__main__':
    main()
