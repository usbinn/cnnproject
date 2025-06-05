import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
import math

# SAM optimizer implementation (Google Research)
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
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

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

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class StochasticDepth(nn.Module):
    """Stochastic Depth for regularization"""
    def __init__(self, drop_rate):
        super(StochasticDepth, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training:
            return x
        
        keep_prob = 1 - self.drop_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, in_channels, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(se_channels, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, 
                 se_ratio=0.25, drop_rate=0.0):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_rate = drop_rate
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU()
            )
        else:
            self.expand = nn.Identity()
            
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride, 
                     kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(expanded_channels, se_ratio)
        
        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Stochastic Depth
        if drop_rate > 0:
            self.stochastic_depth = StochasticDepth(drop_rate)
        else:
            self.stochastic_depth = nn.Identity()

    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.expand(x)
        
        # Depthwise + SE
        x = self.depthwise(x)
        x = self.se(x)
        
        # Pointwise
        x = self.pointwise(x)
        
        # Residual connection with stochastic depth
        if self.use_residual:
            x = self.stochastic_depth(x)
            x = x + identity
            
        return x

class EfficientNetL2_CIFAR(nn.Module):
    """EfficientNet-L2 adapted for CIFAR-100"""
    def __init__(self, num_classes=100, width_mult=1.0, depth_mult=1.0, dropout_rate=0.4):
        super(EfficientNetL2_CIFAR, self).__init__()
        
        # EfficientNet-L2 configuration (adapted for CIFAR)
        # [expand_ratio, channels, repeats, stride, kernel_size]
        settings = [
            [1,  16, 1, 1, 3],  # Stage 1
            [6,  24, 2, 2, 3],  # Stage 2  
            [6,  40, 2, 2, 5],  # Stage 3
            [6,  80, 3, 2, 3],  # Stage 4
            [6, 112, 3, 1, 5],  # Stage 5
            [6, 192, 4, 2, 5],  # Stage 6
            [6, 320, 1, 1, 3],  # Stage 7
        ]
        
        # Stem
        stem_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 1, 1, bias=False),  # stride=1 for CIFAR
            nn.BatchNorm2d(stem_channels),
            nn.SiLU()
        )
        
        # Build blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        total_blocks = sum([int(math.ceil(repeats * depth_mult)) for _, _, repeats, _, _ in settings])
        block_idx = 0
        
        for expand_ratio, channels, repeats, stride, kernel_size in settings:
            out_channels = int(channels * width_mult)
            repeats = int(math.ceil(repeats * depth_mult))
            
            for i in range(repeats):
                # Stochastic depth rate increases linearly
                drop_rate = dropout_rate * block_idx / total_blocks
                
                self.blocks.append(MBConvBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    se_ratio=0.25,
                    drop_rate=drop_rate
                ))
                
                if i == 0:
                    in_channels = out_channels
                block_idx += 1
        
        # Head
        head_channels = int(1280 * width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(head_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def create_effnet_l2_cifar(num_classes=100):
    """Create EfficientNet-L2 model for CIFAR-100"""
    # EfficientNet-L2 scaling: width=4.3, depth=5.3
    # Reduced for CIFAR-100 to avoid overfitting
    return EfficientNetL2_CIFAR(
        num_classes=num_classes,
        width_mult=2.0,    # Reduced from 4.3
        depth_mult=2.0,    # Reduced from 5.3
        dropout_rate=0.5
    )

def get_cifar100_loaders(batch_size=128, num_workers=4):
    """Create CIFAR-100 data loaders with advanced augmentation"""
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    
    # Training augmentations (inspired by EfficientNet training)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),  # RandAugment
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    # Validation transforms
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transforms
    )
    
    val_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_transforms
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

def train_with_sam(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                   device, epochs=150, patience=20):
    """Train model with SAM optimizer"""
    best_acc = 0.0
    no_improve = 0
    os.makedirs("checkpoints", exist_ok=True)
    
    print("🚀 Starting training with SAM...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # SAM first step: ascent to find adversarial weights
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # SAM second step: descent with adversarial weights
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
            # Statistics
            with torch.no_grad():
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
                running_loss += loss.item() * inputs.size(0)
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Validation phase
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
            }, "effnet_l2_sam_best.pth")
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
    model = create_effnet_l2_cifar(num_classes=100).to(device)
    
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
    
    # Create model
    model = create_effnet_l2_cifar(num_classes=100).to(device)
    print(f"📊 Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Data loaders
    train_loader, val_loader = get_cifar100_loaders(batch_size=64, num_workers=4)
    print(f"📚 Data loaders created (Train: {len(train_loader)}, Val: {len(val_loader)})")
    
    # Loss and optimizer (SAM configuration from paper)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SAM(model.parameters(), AdamW, lr=0.001, weight_decay=0.01, eps=1e-8, rho=0.05)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=150, eta_min=1e-6)
    
    print("🎯 Training configuration:")
    print(f"   - Optimizer: SAM + AdamW (rho=0.05)")
    print(f"   - Learning rate: 0.001 → 1e-6 (Cosine)")
    print(f"   - Weight decay: 0.01")
    print(f"   - Label smoothing: 0.1")
    print(f"   - Batch size: 64")
    
    # Train model
    best_acc = train_with_sam(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs=150, patience=25
    )
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_acc:.2f}%")
    
    # Uncomment for inference
    # test_dir = "./Dataset/CImages"
    # model_path = "effnet_l2_sam_best.pth"
    # output_file = "result_effnet_l2_sam.txt"
    # inference_and_save(model_path, test_dir, output_file)

if __name__ == '__main__':
    main() 