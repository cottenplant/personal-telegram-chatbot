from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from chatbot.config.settings import paths_local

import pickle
import random
import pandas as pd


def pick_random_seed_text(text_sequences):
    random.seed(101)
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

    # 15 words, padding, truncating pre-
    input_text = seed_text

    for i in range(num_gen_words):
        # take input text string, encode to sequence
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]

        # make sure we pad up to our trained rate (60?)
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')

        # predict class probabilities for each word (returns index)
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]

        # grab word
        pred_word = tokenizer.index_word[pred_word_ind]

        # update the sequence of input text (shifting one over with the new word)
        input_text += ' ' + pred_word

        output_text.append(pred_word)

    # Make it look like a sentence.
    return ' '.join(output_text)


def main():
    # load models and dependencies
    model = load_model(paths_local['model_path'])
    with open(paths_local['tokenizer_path'], 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # set random seed
    text_sequences_df = pd.read_csv(paths_local['text_sequences'])
    random_seed_text = pick_random_seed_text(text_sequences_df)
    
    # generate some wisdom
    words_of_wisdom = []
    for _ in range(100):
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            seq_len=paths_local['seq_len'],
            seed_text=random_seed_text,
            num_gen_words=paths_local['num_gen_words']
        )
        words_of_wisdom.append(result)
    print(words_of_wisdom)
    pd.DataFrame(words_of_wisdom).to_csv(paths_local['generated_csv'], index=False)
