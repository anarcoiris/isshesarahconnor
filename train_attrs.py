# train_attrs.py
import os
import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIG (match labels in app_features.py) ---
HAIR_COLORS = ["black", "brown", "blonde", "red", "gray", "other"]
EYE_COLORS  = ["brown", "blue", "green", "other"]
ACCESSORIES = ["glasses", "hat", "earrings"]

NUM_HAIR = len(HAIR_COLORS)
NUM_EYE = len(EYE_COLORS)
NUM_ACC = len(ACCESSORIES)

# ---------------- Dataset ----------------
class AttrsDataset(Dataset):
    """
    CSV expected columns:
      image_path, hair (int index), eye (int index), glasses (0/1), hat (0/1), earrings (0/1)
    Example row:
      /path/to/img.jpg, 1, 0, 1, 0, 0
    """
    def __init__(self, df: pd.DataFrame, root: str = ".", transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        hair = int(row['hair'])
        eye  = int(row['eye'])
        accs = torch.tensor([int(row.get(k, 0)) for k in ACCESSORIES], dtype=torch.float32)
        return img, hair, eye, accs

# ---------------- Model ----------------
class MultiAttrModel(nn.Module):
    def __init__(self, n_hair=NUM_HAIR, n_eye=NUM_EYE, n_acc=NUM_ACC, pretrained=True):
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.hair_head = nn.Linear(in_features, n_hair)
        self.eye_head  = nn.Linear(in_features, n_eye)
        self.acc_head  = nn.Linear(in_features, n_acc)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "hair": self.hair_head(feat),
            "eye": self.eye_head(feat),
            "acc": self.acc_head(feat),
        }

# ---------------- Training utils ----------------
def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    hairs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    eyes  = torch.tensor([b[2] for b in batch], dtype=torch.long)
    accs  = torch.stack([b[3] for b in batch])
    return imgs, hairs, eyes, accs

def evaluate(model, loader, device):
    model.eval()
    total = 0
    hair_correct = 0
    eye_correct = 0
    acc_losses = []
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for imgs, hairs, eyes, accs in loader:
            imgs = imgs.to(device)
            hairs = hairs.to(device)
            eyes = eyes.to(device)
            accs = accs.to(device)
            out = model(imgs)
            hair_pred = torch.argmax(out['hair'], dim=1)
            eye_pred  = torch.argmax(out['eye'], dim=1)
            hair_correct += (hair_pred == hairs).sum().item()
            eye_correct  += (eye_pred == eyes).sum().item()
            # loss values for acc only (not used for training here)
            acc_losses.append(bce(out['acc'], accs).item())
            total += imgs.size(0)
    return {
        "hair_acc": hair_correct / total if total else 0.0,
        "eye_acc": eye_correct / total if total else 0.0,
        "acc_bce": sum(acc_losses) / len(acc_losses) if acc_losses else 0.0
    }

# ---------------- Train loop ----------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.csv)
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=42, stratify=df['hair'] if 'hair' in df.columns else None)

    transform_train = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = AttrsDataset(train_df, root=args.root, transform=transform_train)
    val_ds   = AttrsDataset(val_df,   root=args.root, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.workers)

    model = MultiAttrModel(pretrained=True).to(device)

    # Optionally freeze backbone initially
    if args.freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    best_val_score = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, hairs, eyes, accs in pbar:
            imgs = imgs.to(device)
            hairs = hairs.to(device)
            eyes = eyes.to(device)
            accs = accs.to(device)

            out = model(imgs)

            loss_hair = ce(out['hair'], hairs)
            loss_eye  = ce(out['eye'], eyes)
            loss_acc  = bce(out['acc'], accs)

            loss = args.w_hair * loss_hair + args.w_eye * loss_eye + args.w_acc * loss_acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=running_loss / ((pbar.n+1)*imgs.size(0)))

        # unfreeze backbone after freeze_backbone_epochs
        if epoch == args.freeze_backbone_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr/10, weight_decay=1e-4)

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_score = val_metrics['hair_acc'] + val_metrics['eye_acc']  # simple combined metric
        print(f"Epoch {epoch} summary: val hair acc {val_metrics['hair_acc']:.4f}, eye acc {val_metrics['eye_acc']:.4f}, acc bce {val_metrics['acc_bce']:.4f}")

        # Save best
        if val_score > best_val_score:
            best_val_score = val_score
            out_path = Path(args.out_dir) / "model_attrs.pth"
            torch.save(model.state_dict(), out_path)
            print(f"Saved best model to {out_path} (score {best_val_score:.4f})")

    print("Training finished.")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="CSV with image_path, hair, eye, glasses, hat, earrings")
    parser.add_argument("--root", type=str, default=".", help="root dir for images")
    parser.add_argument("--out-dir", type=str, default=".", help="where to save model_attrs.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--w-hair", type=float, default=1.0)
    parser.add_argument("--w-eye", type=float, default=1.0)
    parser.add_argument("--w-acc", type=float, default=1.0)
    args = parser.parse_args()
    train(args)
