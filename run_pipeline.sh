#!/usr/bin/env bash
set -e

POS="$1"
NEG="$2"
NUM=${3:-300}

if [ -z "$POS" ] || [ -z "$NEG" ]; then
  echo "Usage: ./run_pipeline.sh \"positive topic\" \"negative topic\" [num_images]"
  exit 1
fi

echo "Downloading images for '$POS' and '$NEG' ($NUM images each)"
python3 scripts/download_images.py --pos "$POS" --neg "$NEG" --num $NUM --engine google

echo "Preparing dataset (clean + split)"
python3 scripts/prepare_dataset.py --raw data/raw --out data/final --train 0.7 --val 0.15 --test 0.15

echo "Training model"
python3 scripts/train.py --data data/final --epochs 12 --batch 32 --lr 1e-4

echo "Evaluating"
python3 scripts/evaluate.py --data data/final --model models/best_model.pth --out reports

echo "Exporting model"
python3 scripts/export.py --model models/best_model.pth --out models

echo "Pipeline finished"
