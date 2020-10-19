#!usr/bin/env python3

export PYTHONPATH=${pwd} && python scripts/parse_transcripts.py
export PYTHONPATH=${pwd} && python scripts/tokenize_train_generate.py

echo "Done!"
