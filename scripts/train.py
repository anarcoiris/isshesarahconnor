#!/usr/bin/env python3
"""
Entrenamiento con transfer learning y AMP.
Uso básico:
  python scripts/train.py --data data/final --epochs 15 --batch 32 --lr 1e-4
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score


def make_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_transform, val_transform


def build_model(num_classes, pretrained=True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
    return running_loss


def eval_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return val_loss, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/final')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--models_dir', default='models')
    parser.add_argument('--freeze_epochs', type=int, default=3, help='epochs to train only head')
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.models_dir, exist_ok=True)

    train_transform, val_transform = make_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.num_classes, pretrained=True)
    model = model.to(device)

    # Freeze backbone except fc
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=args.lr/10)
            scaler = GradScaler()
            print('Unfroze backbone, lower lr for fine-tuning')

        running_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        epoch_loss = running_loss / len(train_loader.dataset)

        val_loss, y_true, y_pred = eval_model(model, val_loader, criterion, device)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} - train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)")

        # Save checkpoint
        ckpt = os.path.join(args.models_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.models_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print('Saved best model to', best_path)

        scheduler.step(val_loss)

if __name__ == '__main__':
    main()
