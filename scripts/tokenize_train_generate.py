# Should split into preprocess and train so can use vars
import numpy as np
import pandas as pd
import pickle
import random
import spacy

from scripts.settings import RelativePath

from tensorflow import keras
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
    pd.DataFrame(sequences).to_csv(path_local.sequences_seed_path, index=False)
    
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


def create_model(vocabulary_size, seq_len, multiplier):
    model = Sequential()
    
    # definining input dimension for Embedding = entirety of vocab
    # definining output dimension = sequence length
    # defining input_length = sequence length
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    
    # more neurons, 2x sequence length?
    model.add(LSTM(units=seq_len*multiplier, return_sequences=True))
    model.add(LSTM(units=seq_len*multiplier))
    model.add(Dense(units=seq_len*multiplier, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax'))

    # 'categorical_crossentropy' means we are treating each vocabulary word as its own category
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def create_fit_model():
    # read in preprocess and create token sequences
    message_text = read_file(path_local.chats_processed_txt)
    tokenized_text = preprocess(message_text)
    text_sequences = create_token_sequence(tokenized_text)
    
    # tokenize data using token sequences in keras
    tokenizer, sequences = keras_tokenizer(text_sequences)
    
    # train, test, split
    vocab_size, sequence_length, X, y = train_test_split(tokenizer, sequences)
    
    # create model - add 1 space to hold 0 for padding
    model = create_model(vocab_size+1, path_local.seq_len, path_local.multiplier)

    # train model
    model.fit(X, y, batch_size=128, epochs=200, verbose=1)

    # save pickle model
    model.save(
        path_local.model_path, 
        overwrite=True, 
        include_optimizer=True, 
        save_format=None, 
        signatures=None, 
        options=None,
    )
    # not sure how to save tokenizer
    # pickle.dump(tokenizer, open(path_local.tokenizer_path), 'wb'))

    return model, tokenizer


def pick_random_seed_text():
    # can use anything really
    seed_df = pd.read_csv(path_local.sequences_seed_path)
    
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
    model, tokenizer = create_fit_model()
    random_seed_text = pick_random_seed_text()

    generate_text(
        model=model,
        tokenizer=tokenizer, 
        seq_len=path_local.seq_len,
        seed_text=random_seed_text, 
        num_gen_words=path_local.num_gen_words
    )


if __name__ == '__main__':
    path_local = RelativePath(
        user='samco', 
        seq_length=15,
        multiplier=4,
        num_gen_words=25,
        num_epochs=200
    )
    main()
