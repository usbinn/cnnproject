import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_cifar100_custom_split(batch_size=128, train_ratio=0.8):
    """CIFAR-100을 직접 80:20으로 분할하는 함수"""
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    
    # Training augmentations
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 방법 1: 원본 train set(50,000)을 80:20으로 분할
    full_train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=None  # transform는 나중에
    )
    
    # 80:20 분할 계산
    total_size = len(full_train_dataset)  # 50,000
    train_size = int(train_ratio * total_size)  # 40,000
    val_size = total_size - train_size  # 10,000
    
    print(f"📊 Custom split:")
    print(f"   Total: {total_size:,}")
    print(f"   Train: {train_size:,} ({train_size/total_size*100:.1f}%)")
    print(f"   Val: {val_size:,} ({val_size/total_size*100:.1f}%)")
    
    # 랜덤 분할 (재현 가능하게 seed 고정)
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(total_size), [train_size, val_size], generator=generator
    )
    
    # 인덱스를 실제 리스트로 변환
    train_idx = train_indices.indices
    val_idx = val_indices.indices
    
    # Subset 생성 (transform 적용)
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_idx)
    
    # Transform 적용을 위한 커스텀 Dataset 클래스
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __len__(self):
            return len(self.subset)
            
        def __getitem__(self, idx):
            image, label = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
    
    # Transform 적용
    train_dataset_transformed = TransformedDataset(train_dataset, train_transforms)
    val_dataset_transformed = TransformedDataset(val_dataset, val_transforms)
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset_transformed, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset_transformed, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader

def get_cifar100_standard_split(batch_size=128):
    """표준 CIFAR-100 분할 (현재 방식)"""
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 표준 분할: train=True (50k), train=False (10k)
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transforms
    )
    
    val_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_transforms
    )
    
    print(f"📊 Standard split:")
    print(f"   Train: {len(train_dataset):,} (83.3%)")
    print(f"   Val: {len(val_dataset):,} (16.7%)")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader

def compare_splits():
    """두 분할 방식 비교"""
    print("🔍 CIFAR-100 분할 방식 비교\n")
    
    # 방법 1: 표준 분할
    print("1️⃣ 표준 분할 (현재 방식)")
    train_loader1, val_loader1 = get_cifar100_standard_split(batch_size=64)
    
    print()
    
    # 방법 2: 커스텀 분할
    print("2️⃣ 커스텀 80:20 분할")
    train_loader2, val_loader2 = get_cifar100_custom_split(batch_size=64, train_ratio=0.8)
    
    print("\n" + "="*50)
    print("💡 어떤 방식을 선택할까?")
    print("   - 표준 분할: 논문 비교용, 더 많은 학습 데이터")
    print("   - 커스텀 분할: 정확히 80:20, 동일한 데이터 소스")

if __name__ == "__main__":
    compare_splits() 