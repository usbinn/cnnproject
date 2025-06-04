import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ------------------- 모델 정의 -------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ------------------- 데이터 로딩 -------------------
def load_test_images(test_dir):
    image_paths = [os.path.join(test_dir, fname)
                  for fname in os.listdir(test_dir)
                  if fname.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort numerically by filename (without extension)
    image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

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
            formatted_number = f"{int(number):04d}"
            f.write(f"{formatted_number}, {pred:02d}\n")

    print(f"Saved inference results to {output_file}")

if __name__ == '__main__':
    test_dir = "./Dataset/CImages"
    model_path = "weight_가반1조_0602_1410.pth"
    output_file = "result_가반1조_0602_1410.txt"
    inference_and_save(model_path, test_dir, output_file)