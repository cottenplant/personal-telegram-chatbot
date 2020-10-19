import numpy as np
import pandas as pd
import pickle
import random
import spacy

from chatbot.config.settings import paths_local

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences


def read_file(filepath):
    with open(filepath) as infile:
        str_text = infile.read()

    return str_text


def preprocess(message_text):
    excluded_chars = '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '
    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

    return [token.text.lower() for token in nlp(message_text)
            if token.text not in excluded_chars]


def make_tokenized_sequence(tokens, train_cycle=15):
    # because we have such short sentences, should predict less
    train_length = train_cycle + 1
    text_sequences = []

    for i in range(train_length, len(tokens)):
        seq = tokens[i-train_length:i]
        text_sequences.append(seq)

    return text_sequences


def fit_tokenizer_on_text(text_sequences):
    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    sequences = tokenizer.texts_to_sequences(text_sequences)
    # index_word tokenized with UID, word counts, vocabulary size - interesting
    # for i in sequences[0]:
    #     print(f'UUID {i} : {tokenizer.index_word[i]}')
    # print(f'word counts: {tokenizer.word_counts}')
    # print(f'vocab size: {len(tokenizer.word_counts)}')

    return sequences, tokenizer


def split_data_fit_model(sequences, vocab_size, multiplier):    
    # pull all columns, split last to predict
    sequence_array = np.array(sequences)
    X = sequence_array[:, :-1]
    y = sequence_array[:, -1]
    y = to_categorical(y, num_classes=vocab_size + 1)
    seq_len = X.shape[1]
    
    # instantiate model
    model = Sequential()
    model.add(Embedding(vocab_size, seq_len, input_length=seq_len))
    model.add(LSTM(units=seq_len*multiplier, return_sequences=True))
    model.add(LSTM(units=seq_len*multiplier))
    model.add(Dense(units=seq_len*multiplier, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    
    # 'categorical_crossentropy' means we are treating each vocabulary word as its own category
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    # create and train model - add 1 space to hold 0 for padding
    model.fit(X, y, batch_size=128, epochs=paths_local['num_epochs'], verbose=1)
    
    return model, seq_len


def pickle_model_tokenizer(model, tokenizer):
    # save model
    model.save(
        paths_local['model_path'],
        overwrite=True,
        include_optimizer=True)
    # save tokenizer
    pickle.dump(
        tokenizer, 
        open(paths_local['tokenizer_path'], 'wb'))


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
    for _ in range(num_gen_words):
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
    # read in preprocess and create token sequences
    message_text = read_file(paths_local['chats_processed_txt'])
    preprocessed_text = preprocess(message_text)
    text_sequences = make_tokenized_sequence(preprocessed_text, train_cycle=15)
    
    # save text sequences for later
    pd.DataFrame(text_sequences).to_csv(paths_local['text_sequences'], index=False)
    sequences, tokenizer = fit_tokenizer_on_text(text_sequences)
    
    # calculate vocav size create model, and fit data
    vocab_size = len(tokenizer.word_counts)
    model, seq_len = split_data_fit_model(sequences, vocab_size, multiplier=4)
    
    # fit and pickle model for later use
    pickle_model_tokenizer(model, tokenizer)
    seed_text = pick_random_seed_text(text_sequences)
    
    # test-run and generate some wisdom
    words_of_wisdom = []
    for _ in range(100):
        result = generate_text(
            model,
            tokenizer,
            seq_len,
            seed_text,
            num_gen_words=paths_local['num_gen_words']
        )
        words_of_wisdom.append(result)

    # print and save some samples for fine-tuning
    print(words_of_wisdom)
    pd.DataFrame(words_of_wisdom).to_csv(paths_local['generated_csv'], index=False)


if __name__ == '__main__':
    main()
