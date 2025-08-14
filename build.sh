#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt

# Optional model pre-download
python -c "
try:
    from transformers import pipeline
    print('Loading models...')
    pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')
    pipeline('summarization', model='sshleifer/distilbart-cnn-6-6')
except Exception as e:
    print(f'Model pre-loading skipped: {str(e)}')
"
