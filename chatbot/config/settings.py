from pathlib import Path

paths_local = dict(
    user='samco',
    seq_length=15,
    multiplier=4,
    num_gen_words=25,
    num_epochs=200,
    chats_raw_json=Path('./data/input/result.json'),
    sequences_seed_path=Path('./models/sequences_seed_text.csv'),
    chats_processed_json=Path('./data/output/samco.json'),
    chats_processed_txt=Path('./data/output/samco.txt'),
    model_path=Path('./models/chatbot_model_200.h5'),
    tokenizer_path=Path('./models/chatbot_tokenizer_200')
)
