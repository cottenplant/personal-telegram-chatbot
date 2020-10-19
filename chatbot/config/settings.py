from pathlib import Path

paths_local = dict(
    user='samco',
    seq_len=15,
    multiplier=4,
    num_gen_words=25,
    num_epochs=300,
    chats_raw_json=Path('./data/input/result.json'),
    text_sequences=Path('./models/text_sequences.txt'),
    chats_processed_json=Path('./data/output/samco.json'),
    chats_processed_txt=Path('./data/output/samco.txt'),
    generated_csv=Path('./data/output/chatbot.csv'),
    model_path=Path('./models/chatbot_model_200.h5'),
    tokenizer_path=Path('./models/chatbot_tokenizer_200.pickle')
)
