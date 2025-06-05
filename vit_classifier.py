import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import create_transform
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from PIL import Image, ImageOps, ImageEnhance
import os
import requests
from tqdm import tqdm

class TrivialAugment:
    def __init__(self, num_ops=2, magnitude=10):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.ops = [
            ('identity', 0, 1),
            ('auto_contrast', 0, 1),
            ('equalize', 0, 1),
            ('rotate', -30, 30),
            ('solarize', 0, 256),
            ('color', 0.1, 1.9),
            ('posterize', 4, 8),
            ('contrast', 0.1, 1.9),
            ('brightness', 0.1, 1.9),
            ('sharpness', 0.1, 1.9),
            ('shear_x', -0.3, 0.3),
            ('shear_y', -0.3, 0.3),
            ('translate_x', -0.5, 0.5),
            ('translate_y', -0.5, 0.5)
        ]

    def __call__(self, img):
        for _ in range(self.num_ops):
            op_name, min_val, max_val = random.choice(self.ops)
            magnitude = random.uniform(0, self.magnitude)
            if op_name == 'identity':
                continue
            elif op_name == 'auto_contrast':
                img = ImageOps.autocontrast(img)
            elif op_name == 'equalize':
                img = ImageOps.equalize(img)
            elif op_name == 'rotate':
                angle = min_val + (max_val - min_val) * magnitude / self.magnitude
                img = img.rotate(angle)
            elif op_name == 'solarize':
                threshold = min_val + (max_val - min_val) * magnitude / self.magnitude
                img = ImageOps.solarize(img, threshold)
            elif op_name in ['color', 'contrast', 'brightness', 'sharpness']:
                factor = min_val + (max_val - min_val) * magnitude / self.magnitude
                if op_name == 'color':
                    img = ImageEnhance.Color(img).enhance(factor)
                elif op_name == 'contrast':
                    img = ImageEnhance.Contrast(img).enhance(factor)
                elif op_name == 'brightness':
                    img = ImageEnhance.Brightness(img).enhance(factor)
                elif op_name == 'sharpness':
                    img = ImageEnhance.Sharpness(img).enhance(factor)
            elif op_name in ['shear_x', 'shear_y', 'translate_x', 'translate_y']:
                magnitude = min_val + (max_val - min_val) * magnitude / self.magnitude
                if op_name == 'shear_x':
                    img = img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
                elif op_name == 'shear_y':
                    img = img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
                elif op_name == 'translate_x':
                    img = img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0], 0, 1, 0))
                elif op_name == 'translate_y':
                    img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1]))
        return img

class SoftAugmentation:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def __call__(self, img):
        if random.random() < self.alpha:
            # Random color jitter
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(img)
            # Random horizontal flip
            if random.random() < 0.5:
                img = transforms.RandomHorizontalFlip()(img)
            # Random rotation
            if random.random() < 0.5:
                img = transforms.RandomRotation(15)(img)
        return img

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p not in self.state:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`")

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        TrivialAugment(num_ops=2, magnitude=10),
        SoftAugmentation(alpha=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
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

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(val_loader), 100. * correct / total

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_transform, val_transform = get_transforms()
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Create model directory if it doesn't exist
    os.makedirs('model_weights', exist_ok=True)
    weights_path = 'model_weights/vit_base_patch16_224.pth'

    # Download weights if they don't exist
    if not os.path.exists(weights_path):
        print("Downloading pretrained weights...")
        url = "https://huggingface.co/timm/vit_base_patch16_224/resolve/main/model.safetensors"
        download_file(url, weights_path)
        print("Weights downloaded successfully!")

    # Create model
    print("Creating model...")
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=10,
        in_chans=3
    )
    
    # Load pretrained weights
    print("Loading pretrained weights...")
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict, strict=False)
    print("Model created and weights loaded successfully!")

    model = model.to(device)
    print(f"Model moved to {device}")

    # Create optimizer and scheduler
    base_optimizer = AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0
    for epoch in range(100):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()

        print(f'Epoch: {epoch+1}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')
            print(f'New best model saved! Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main() 