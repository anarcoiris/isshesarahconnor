import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os

# Dataset personalizado para imágenes de Sarah Connor
class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name)
        label = 0 if 'sarah' in img_name else 1  # 'sarah' o 'no_sarah' en el nombre del archivo
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transformaciones para preprocesar las imágenes
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datos
dataset = CustomDataset(img_dir='./data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Modelo preentrenado (ResNet50)
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 clases: Sarah Connor vs Otros

# Configuración para entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Entrenamiento
for epoch in range(10):  # 10 épocas de entrenamiento
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "sarah_connor_model.pth")
