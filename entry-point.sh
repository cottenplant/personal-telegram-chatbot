#!usr/bin/env python3

export PYTHONPATH='.' && python chatbot/parse_transcripts.py
export PYTHONPATH='.' && python chatbot/tokenize_train_fit.py
export PYTHONPATH='.' && python chatbot/generate_text.py
