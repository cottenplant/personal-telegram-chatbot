def main():
    # read in preprocess and create token sequences
    message_text = read_file(paths_local['chats_processed_txt'])
    preprocessed_text = preprocess(message_text)
    text_sequences = make_tokenized_sequence(tokenized_text, train_cycle=15)
    
    # save text sequences for later
    pd.DataFrame(text_sequences).to_csv(paths_local['text_sequences'], index=False)
		sequences, tokenizer = fit_tokenizer_on_text(test_sequences)
		
		print(sequences)
		print(tokenizer)
    
    vocab_size = len(tokenizer.word_counts)
    model = create_model(vocab_size, seq_len, multiplier):

    model = fit_model(tokenizer, sequences)
    pickle_model_tokenizer(model, tokenizer)
    seed_test_random = pick_random_seed_text(text_sequences)

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


if __name__ == '__main__':
		main()

