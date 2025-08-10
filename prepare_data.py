import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import pickle
import os

def main():
    download_nltk_resources()

    DATA_FOLDER = 'data/'
    INPUT_FILE_PATH = DATA_FOLDER + 'chat_labeled.csv'
    TEXT_COLUMN_NAME = 'key'
    LABEL_COLUMN_NAME = 'sentiment'

    df = pd.read_csv(INPUT_FILE_PATH)
    df.dropna(subset=[TEXT_COLUMN_NAME, LABEL_COLUMN_NAME], inplace=True)

    sentences = df[TEXT_COLUMN_NAME].apply(
        lambda msg: word_tokenize(re.sub(r'^a-zA-Zа-яА-Я\s', '', str(msg).lower()))
    ).tolist()

    label_map = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
    labels = df[LABEL_COLUMN_NAME].map(label_map).values

    VOCAB_SIZE = 10000
    MAX_LENGTH = 30

    word_to_idx_map = build_vocab(sentences, vocab_size=VOCAB_SIZE)

    vocab_path = os.path.join(DATA_FOLDER, 'word_to_idx.pickle')
    with open(vocab_path, 'wb') as handle:
        pickle.dump(word_to_idx_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Vocab was saved in "{vocab_path}"')

    processed_X = vectorize_and_pad(sentences, word_to_idx_map, max_length=MAX_LENGTH)
    processed_y = labels

    print('Example of processing:')
    print(f'Original: "{df[TEXT_COLUMN_NAME].iloc[0]}"')
    print(f'Tokens: {sentences[0]}')
    print(f'Aligned sequence (length {len(processed_X[0])}):\n{processed_X[0]}')
    
    np.save(os.path.join(DATA_FOLDER, 'processed_X.npy'), processed_X)
    np.save(os.path.join(DATA_FOLDER, 'processed_y.npy'), processed_y)
    print('\nReady for training data was saved')

def download_nltk_resources():
    resources = {'tokenizers/punkt': 'punkt'}
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
            print(f'Resource "{resource_name}" is already installed.')
        except LookupError:
            print(f'Resource "{resource_name}" not found. Installing...')
            nltk.download(resource_name, quiet=True)
            print(f'Installing "{resource_name}" was complete.')

def build_vocab(sentences, vocab_size=10000):
    print('Building vocab...')
    all_words = [word for sentence in sentences for word in sentence]
    
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(vocab_size - 2)

    word_to_idx = {word: i+2 for i, (word, _) in enumerate(most_common_words)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1

    print(f'Vocab was built. Size: {len(word_to_idx)} words.')
    return word_to_idx

def vectorize_and_pad(sentences, word_to_idx, max_length=30):
    print('Vectorization and data alignment...')
    sequences = []
    for sentence in sentences:
        sequence = [word_to_idx.get(word, 1) for word in sentence]
        sequences.append(sequence)

    padded_sequences = np.zeros((len(sequences), max_length), dtype=np.int64)
    for i, sequence in enumerate(sequences):
        sequence = sequence[:max_length]
        padded_sequences[i, :len(sequence)] = sequence

    print('Data is ready.')
    return padded_sequences


if __name__ == '__main__':
    main()