import numpy as np
import pickle
import spacy

from nlp.config.settings import paths_local

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


def read_file(filepath):
    with open(filepath) as infile:
        str_text = infile.read()

    return str_text


def separate_punc(message_text):
    excluded_chars = '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '
    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

    return [token.text.lower() for token in nlp(message_text)
            if token.text not in excluded_chars]


def create_sequence_tokens(tokens, train_cycle=25):
    # because we have such short sentences, should predict less
    train_len = train_cycle + 1
    text_sequences = []
    for i in range(train_len, len(tokens)):
        seq = tokens[i-train_len:i]
        text_sequences.append(seq)
    print(f"\nlen_text_sequences:\n{len(text_sequences)}")

    return text_sequences


def keras_tokenization(text_sequences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    sequences = tokenizer.texts_to_sequences(text_sequences)
    vocabulary_size = len(tokenizer.word_counts)

    return sequences, tokenizer, vocabulary_size


def create_model(vocabulary_size, seq_len=25):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def train_test_split(sequences, vocabulary_size):
    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = to_categorical(y, num_classes=vocabulary_size+1)
    seq_len = X.shape[1]
    print(f"\nseq_len:\n{seq_len}")

    return X, y, seq_len


def main():
    message_text = read_file(paths_local['chats_processed_txt'])
    tokens = separate_punc(message_text)
    text_sequences = create_sequence_tokens(tokens, train_cycle=25)

    sequences, tokenizer, vocabulary_size = keras_tokenization(text_sequences)
    sequences = np.array(sequences)

    X, y, seq_len = train_test_split(sequences, vocabulary_size)
    model = create_model(vocabulary_size+1, seq_len)
    model.fit(X, y, batch_size=128, epochs=320, verbose=1)

    model.save(paths_local['model_path'])
    pickle.dump(tokenizer, open(paths_local['tokenizer_path'], 'wb'))
    pickle.dump(text_sequences, open(paths_local['text_sequences'], 'wb'))
    pickle.dump(tokenizer.word_counts, open(paths_local['tokenizer_wordcounts'], 'wb'))


if __name__ == '__main__':
    main()
