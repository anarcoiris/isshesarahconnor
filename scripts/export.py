#!/usr/bin/env python3
"""
Exporta el modelo a TorchScript y ONNX.
Uso:
  python scripts/export.py --model models/best_model.pth --out models
"""
import argparse
import os
import torch
from torchvision import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/best_model.pth')
    parser.add_argument('--out', default='models')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # TorchScript (trace)
    example = torch.randn(1,3,224,224).to(device)
    traced = torch.jit.trace(model, example)
    traced.save(os.path.join(args.out, 'model_traced.pt'))

    # ONNX
    onnx_path = os.path.join(args.out, 'model.onnx')
    torch.onnx.export(model, example, onnx_path, opset_version=11, input_names=['input'], output_names=['output'])
    print('Exported TorchScript and ONNX to', args.out)

if __name__ == '__main__':
    main()
