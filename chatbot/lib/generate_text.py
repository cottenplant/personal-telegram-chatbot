import random
import pandas as pd
import pickle

from pathlib import Path
from keras.preprocessing.sequence import pad_sequences 
from tensorflow import keras


def pick_random_seed_text():
    # can use anything really
    seed_df = pd.read_csv(seed_path)
    
    random.seed(101)
    random_pick = random.randint(0, len(seed_df))
    random_seed_text = text_to_sequences[random_pick]
    seed_text = ' '.join(random_seed_text)
    print(f'seed_text: {seed_text}')
    
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

    # 15 words, padding, truncating pre-
    input_text = seed_text
    
    for _ in range(num_gen_words):
        # take input text string, encode to sequence
        encoded_text = tokenizer.text_to_sequences([input_text])[0]
        
        # make sure we pad up to our trained rate (60?)
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        
        # predict class probabilities for each word (returns index)
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        
        # grab word
        pre_word = tokenizer.index_word[pred_word_ind]

        # update the sequence of input text (shifting one over with the new word)
        input_text += ' '+pre_word
        output_text.append(pre_word)
    
    # return sentence-like output
    return ' '.join(output_text)


def main():
    # model = keras.models.load_model(model_path)
    with open(model_path, 'wb') as handle:
        model = pickle.load(handle)
    
    with open(tokenizer_path, 'wb') as handle:
        tokenizer = pickle.load(handle)
    
    seq_len = 15
    
    seed_text = pick_random_seed_text()
    
    num_gen_words = 25

    generate_text(
        model,
        tokenizer, 
        seq_len,
        seed_text, 
        num_gen_words
    )


if __name__ == '__main__':
    model_path = Path('./models/chatbot_model_200.h5')
    tokenizer_path = Path('./models/chatbot_tokenizer_200')
    seed_path = Path('./seed/text_sequences.csv')
    main()
