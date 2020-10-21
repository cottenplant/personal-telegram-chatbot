from pathlib import Path


paths_local = dict(
    user='samco',
    chats_raw_json=Path('./data/input/result.json'),
    text_sequences=Path('./models/text_sequences.txt'),
    tokenizer_wordcounts=Path('./models/tokenizer_wordcounts.txt'),
    chats_processed_json=Path('./data/output/samco.json'),
    chats_processed_txt=Path('./data/output/samco.txt'),
    generated_csv=Path('./data/output/chatbot.csv'),
    model_path=Path('./models/chatbot_model.h5'),
    tokenizer_path=Path('./models/chatbot_tokenizer.pickle')
)
