import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
import math

# SAM optimizer implementation
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

# Soft Augmentation Dataset
class SoftAugmentDataset(Dataset):
    """Dataset wrapper that applies soft augmentation with TrivialAugment"""
    def __init__(self, base_dataset, num_classes=100, p_min=0.1, k=2.0):
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.p_min = p_min
        self.k = k
        
        # Base transforms (normalization only)
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # TrivialAugmentWide for training
        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(),  # TrivialAugment instead of RandAugment
            self.base_transform,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
        ])
        
        # No augmentation for some samples
        self.no_augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            self.base_transform
        ])
        
    def __len__(self):
        return len(self.base_dataset)
    
    def _estimate_visibility(self):
        """Estimate visibility based on augmentation strength"""
        # TrivialAugment applies random operations, so we estimate average visibility
        # Strong augmentations like Cutout, aggressive rotations reduce visibility more
        return random.uniform(0.3, 0.9)  # Random visibility for TrivialAugment
    
    def _create_soft_target(self, hard_label, visibility):
        """Create soft target based on visibility"""
        confidence = 1.0 - (1.0 - self.p_min) * ((1.0 - visibility) ** self.k)
        confidence = max(self.p_min, min(1.0, confidence))
        
        soft_target = torch.full((self.num_classes,), (1.0 - confidence) / (self.num_classes - 1))
        soft_target[hard_label] = confidence
        
        return soft_target
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # 80% chance of augmentation, 20% clean
        if random.random() < 0.8:
            image = self.augment_transform(image)
            visibility = self._estimate_visibility()
            soft_target = self._create_soft_target(label, visibility)
            return image, soft_target, True  # True indicates soft target
        else:
            image = self.no_augment_transform(image)
            return image, label, False  # False indicates hard target

# DenseNet implementation
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.bn1(x)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseNet_CIFAR(nn.Module):
    """DenseNet-BC for CIFAR-100"""
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.2, num_classes=100):
        super(DenseNet_CIFAR, self).__init__()
        
        # First convolution (for CIFAR, smaller kernel)
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool5', nn.AdaptiveAvgPool2d(1))
        
        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out

def create_densenet_cifar100():
    """Create DenseNet-121 adapted for CIFAR-100"""
    return DenseNet_CIFAR(
        growth_rate=32,
        block_config=(6, 12, 24, 16),  # DenseNet-121 configuration
        num_init_features=64,
        bn_size=4,
        drop_rate=0.2,
        num_classes=100
    )

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar100_loaders_with_soft_aug(batch_size=128, num_workers=4):
    """Create CIFAR-100 data loaders with TrivialAugment + Soft Augmentation"""
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    
    # Base datasets (no transforms applied yet)
    train_dataset_base = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=None
    )
    
    val_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )
    
    # Apply soft augmentation to training set
    train_dataset = SoftAugmentDataset(train_dataset_base)
    
    # Custom collate function for mixed hard/soft targets
    def collate_fn(batch):
        images = []
        targets = []
        is_soft = []
        
        for image, target, soft_flag in batch:
            images.append(image)
            targets.append(target)
            is_soft.append(soft_flag)
        
        images = torch.stack(images)
        is_soft = torch.tensor(is_soft)
        
        # Separate hard and soft targets
        if any(is_soft):
            # Create mixed batch with proper padding
            max_classes = 100
            batch_targets = []
            for i, (target, soft_flag) in enumerate(zip(targets, is_soft)):
                if soft_flag:
                    batch_targets.append(target)  # Already a soft tensor
                else:
                    # Convert hard label to one-hot
                    hard_target = torch.zeros(max_classes)
                    hard_target[target] = 1.0
                    batch_targets.append(hard_target)
            targets = torch.stack(batch_targets)
        else:
            targets = torch.tensor(targets)
        
        return images, targets, is_soft
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

def train_with_soft_targets(model, train_loader, val_loader, optimizer, scheduler, 
                           device, epochs=200, patience=25):
    """Train model with soft targets and SAM"""
    best_acc = 0.0
    no_improve = 0
    os.makedirs("checkpoints", exist_ok=True)
    
    print("🚀 Starting training with DenseNet + SAM + TrivialAugment + Soft Augmentation...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets, is_soft_batch in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            is_soft_batch = is_soft_batch.to(device)
            
            # Define closure for SAM
            def closure():
                outputs = model(inputs)
                
                # Calculate loss based on target type
                if torch.any(is_soft_batch):
                    # Use KL divergence for soft targets, CE for hard targets
                    loss = 0
                    for i, is_soft in enumerate(is_soft_batch):
                        if is_soft:
                            # Soft target: use KL divergence
                            log_probs = F.log_softmax(outputs[i:i+1], dim=1)
                            soft_target = targets[i:i+1]
                            loss += F.kl_div(log_probs, soft_target, reduction='sum')
                        else:
                            # Hard target: use cross entropy
                            hard_target = targets[i:i+1].argmax(dim=1)
                            loss += F.cross_entropy(outputs[i:i+1], hard_target)
                    loss = loss / len(is_soft_batch)
                else:
                    # All hard targets
                    loss = F.cross_entropy(outputs, targets)
                
                return loss
            
            # SAM optimization
            loss = closure()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            loss = closure()
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
            # Statistics
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = outputs.max(1)
                
                # Get true labels for accuracy calculation
                if torch.any(is_soft_batch):
                    true_labels = targets.argmax(dim=1)
                else:
                    true_labels = targets
                
                correct += preds.eq(true_labels).sum().item()
                total += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Validation
        val_acc = validate(model, val_loader, device)
        
        # Metrics
        epoch_loss = running_loss / total
        train_acc = 100. * correct / total
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, "densenet_sam_soft_best.pth")
            print(f"✅ New best model saved! Acc: {best_acc:.2f}%")
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
        
        scheduler.step()
        print("-" * 50)
    
    return best_acc

def validate(model, val_loader, device):
    """Validate model"""
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

def load_test_images(test_dir):
    """Load test images for inference"""
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

    if images:
        batch = torch.stack(images)
        return batch, filenames
    else:
        return torch.empty(0), []

def inference_and_save(model_path, test_dir, output_file):
    """Perform inference and save results"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return

    # Load model
    model = create_densenet_cifar100().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model from {model_path}")
        print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()

    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found!")
        return

    inputs, filenames = load_test_images(test_dir)
    if len(filenames) == 0:
        print("No images found in test directory!")
        return
        
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()

    with open(output_file, 'w') as f:
        f.write("number, label\n")
        for fname, pred in zip(filenames, preds):
            number = os.path.splitext(fname)[0]
            formatted_number = f"{int(number):04d}"
            f.write(f"{formatted_number}, {pred:02d}\n")

    print(f"Saved inference results to {output_file}")

def main():
    """Main training function"""
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    
    # Create DenseNet model
    model = create_densenet_cifar100().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 DenseNet-121 created with {total_params/1e6:.2f}M parameters")
    
    # Data loaders with soft augmentation
    train_loader, val_loader = get_cifar100_loaders_with_soft_aug(batch_size=64, num_workers=4)
    print(f"📚 Data loaders created (Train: {len(train_loader)}, Val: {len(val_loader)})")
    
    # SAM optimizer
    optimizer = SAM(model.parameters(), AdamW, lr=0.001, weight_decay=0.0005, rho=0.05)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=200, eta_min=1e-6)
    
    print("🎯 Training configuration:")
    print(f"   - Model: DenseNet-121 (Growth Rate: 32)")
    print(f"   - Optimizer: SAM + AdamW (rho=0.05)")
    print(f"   - Learning rate: 0.001 → 1e-6 (Cosine)")
    print(f"   - Weight decay: 0.0005")
    print(f"   - Augmentation: TrivialAugmentWide + Soft Targets")
    print(f"   - Batch size: 64")
    
    # Train model
    best_acc = train_with_soft_targets(
        model, train_loader, val_loader, optimizer, scheduler,
        device, epochs=200, patience=30
    )
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_acc:.2f}%")
    
    # Uncomment for inference
    # test_dir = "./Dataset/CImages"
    # model_path = "densenet_sam_soft_best.pth"
    # output_file = "result_densenet_sam_soft.txt"
    # inference_and_save(model_path, test_dir, output_file)

if __name__ == '__main__':
    main() 