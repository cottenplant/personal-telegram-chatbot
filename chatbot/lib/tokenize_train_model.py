# Should split into preprocess and train so can use vars

import pandas as pd
import spacy
import numpy as np

from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from pickle import dump, load


def read_file(filepath):
    with open(input_file_path) as infile:
        str_text = infile.read()
    
    return str_text


def preprocess(message_text):
    excluded_chars = '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '
    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    
    return [token.text.lower() for token in nlp(message_text) \
            if token.text not in excluded_chars]


def create_token_sequence(tokens, train_cycle=15):
    # because we have such short sentences, should predict less
    train_length = train_cycle + 1
    text_sequences = []
    for i in range(train_length, len(tokens)):
        seq = tokens[i-train_length:i]
        text_sequences.append(seq)
    
    return text_sequences


def keras_tokenizer(text_sequences):
    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    sequences = tokenizer.texts_to_sequences(text_sequences)
    
    # save for random seeding later
    pd.DataFrame(sequences).to_csv(Path('./seed/text_sequences.csv', index=False))
    
    # index_word tokenized with UID, word counts, vocabulary size - interesting
    # for i in sequences[0]:
    #     print(f'UUID {i} : {tokenizer.index_word[i]}')
    # print(f'word counts: {tokenizer.word_counts}')
    # print(f'vocab size: {len(tokenizer.word_counts)}')
    
    return tokenizer, sequences


def train_test_split(tokenizer, sequences):
    vocabulary_size = len(tokenizer.word_counts)
    sequence_array = np.array(sequences)
    
    # pull all columns, split last to predict
    X = sequence_array[:,:-1]
    y = sequence_array[:,-1]
    y = to_categorical(y, num_classes=vocabulary_size+1)

    print(X.shape)
    seq_len = X.shape[1]

    return vocabulary_size, seq_len, X, y


def create_model(vocabulary_size, sequence_length, multiplier):
    model = Sequential()
    
    # definining input dimension for Embedding = entirety of vocab
    # definining output dimension = sequence length
    # defining input_length = sequence length
    model.add(Embedding(vocabulary_size, sequence_length, input_length=sequence_length))
    
    # more neurons, 2x sequence length?
    model.add(LSTM(units=sequence_length*multiplier, return_sequences=True))
    model.add(LSTM(units=sequence_length*multiplier))
    model.add(Dense(units=sequence_length*multiplier, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax'))

    # 'categorical_crossentropy' means we are treating each vocabulary word as its own category
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def main():
    # read in preprocess and and create token sequences
    message_text = read_file(input_file_path)
    tokenized_text = preprocess(message_text)
    text_sequences = create_token_sequence(tokenized_text)
    
    # tokenize data using token sequences in keras
    tokenizer, sequences = keras_tokenizer(text_sequences)
    
    # train, test, split
    vocabulary_size, sequence_length, X, y = train_test_split(tokenizer, sequences)
    
    # create model - add 1 space to hold 0 for padding
    model = create_model(vocabulary_size+1, sequence_length, 4)

    # train model
    model.fit(X, y, batch_size=128, epochs=200, verbose=1)

    # pickle
    model.save(model_output_path)
    dump(tokenizer, open(tokenizer_output_path, 'wb'))


if __name__ == '__main__':
    # static vars
    input_file_path = Path('./data/output/samco.txt')
    model_output_path = Path('./models/chatbot_model_200.h5')
    tokenizer_output_path = Path('./models/chatbot_tokenizer_200')

    main()
