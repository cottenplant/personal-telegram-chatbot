from chatbot.config.settings import paths_local

import pickle
import random
import time
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


def unpickle(filepath):
    with open(filepath, 'rb') as fp:
        return pickle.load(fp)


def pick_random_seed_text(text_sequences):
    seed = round(time.time())
    random.seed(seed)
    random_pick = random.randint(0, len(text_sequences))
    random_seed_text = text_sequences[random_pick]
    seed_text = ' '.join(random_seed_text)

    return seed_text


def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    """
    model: trained
    tokenizer: knowledge about vocab and uid number per token (word)
    seq_len: should be equal seed_text
    seed_text: should be equal seq_length (what it was trained on, otherwise padding)
    num_gen_words: number of words to generate
    """
    output_text = []
    input_text = seed_text
    print(f"\nseed_text:\n{seed_text}\n")
    
    for _ in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += ' ' + pred_word
        output_text.append(pred_word)
    
    return ' '.join(output_text)


def main():
    # unpickle model, tokenizer, text sequencer for seed
    model = load_model(paths_local['model_path'])
    tokenizer = unpickle(paths_local['tokenizer_path'])
    text_sequences = unpickle(paths_local['text_sequences'])
    random_seed_text = pick_random_seed_text(text_sequences)

    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        seq_len=25,
        seed_text=random_seed_text,
        num_gen_words=50
    )
    return result


if __name__ == '__main__':
    main()
