import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==== CONFIG ====
data_dir = "datasets"
batch_size = 32
num_epochs = 32
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== TRANSFORMS ====
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),              # resize về đúng input của MobileNetV2
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # chuẩn ImageNet
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== DATASETS ====
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==== MODEL ====
model = models.mobilenet_v2(pretrained=True)   # load MobileNetV2 pretrained ImageNet
# Thay output classifier thành binary (2 class)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

# ==== LOSS & OPTIMIZER ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==== TRAIN LOOP ====
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
    val_correct, val_total = 0, 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {running_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# ==== SAVE MODEL ====
torch.save(model.state_dict(), "mobilenetv2_occupied.pth")
print("✅ Training done, model saved!")
