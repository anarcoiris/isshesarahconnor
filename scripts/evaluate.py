#!/usr/bin/env python3
"""
Evalúa el modelo en el test set y genera métricas + matriz de confusión.
Uso:
  python scripts/evaluate.py --data data/final --model models/best_model.pth --out reports
"""
import argparse
import os
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/final')
    parser.add_argument('--model', default='models/best_model.pth')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', default='reports')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(args.data, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(test_dataset.classes))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=test_dataset.classes, output_dict=True)
    with open(os.path.join(args.out, 'classification_report.json'), 'w') as f:
        import json
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    ticks = np.arange(len(test_dataset.classes))
    plt.xticks(ticks, test_dataset.classes, rotation=45)
    plt.yticks(ticks, test_dataset.classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(args.out, 'confusion_matrix.png'), bbox_inches='tight')
    print('Saved report and confusion matrix to', args.out)

if __name__ == '__main__':
    main()
