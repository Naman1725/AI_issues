#!/bin/bash
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Pre-downloading ML models..."
python -c "
from transformers import pipeline;
print('Downloading classifier model...');
pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli');
print('Downloading summarizer model...');
pipeline('summarization', model='sshleifer/distilbart-cnn-6-6');
"

echo "Build completed!"
