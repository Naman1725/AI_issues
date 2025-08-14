#!/bin/bash
# Exit immediately if any command fails
set -e

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Pre-downloading models ==="
python -c "
try:
    from transformers import pipeline
    print('Downloading classifier...')
    pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')
    print('Downloading summarizer...')
    pipeline('summarization', model='sshleifer/distilbart-cnn-6-6')
except Exception as e:
    print(f'Model pre-loading skipped: {e}')
"
