import pandas as pd
import numpy as np
import sentencepiece as spm

def main():
    INPUT_FILE_PATH = 'chat_labeled_smart.csv'
    TEXT_COLUMN_NAME = 'key'
    LABEL_COLUMN_NAME = 'sentiment'
    TOKENIZER_MODEL_PATH = 'dota_chat_bpe.model' 
    MAX_LENGTH = 30

    print(f'Loading labeled data from "{INPUT_FILE_PATH}"...')
    try:
        df = pd.read_csv(INPUT_FILE_PATH)
        df.dropna(subset=[TEXT_COLUMN_NAME, LABEL_COLUMN_NAME], inplace=True)
    except FileNotFoundError:
        print(f'Error: file {INPUT_FILE_PATH} not found.')
        exit()

    print(f'Loading SentencePiece tokenizer from "{TOKENIZER_MODEL_PATH}"...')
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_MODEL_PATH)
        VOCAB_SIZE = sp.get_piece_size()
        print(f'Tokenizer loaded successfully. Vocab size: {VOCAB_SIZE}')
    except Exception as e:
        print(f'Error: Failed to load tokenizer model. {e}')
        exit()

    label_map = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
    df['sentiment_id'] = df[LABEL_COLUMN_NAME].map(label_map)
    df.dropna(subset=['sentiment_id'], inplace=True)
    labels = df['sentiment_id'].astype(int).values

    print('\nText vectorization with SentencePiece...')
    sequences = df[TEXT_COLUMN_NAME].apply(lambda text: sp.encode_as_ids(str(text))).tolist()
    
    print('Padding...')
    padded_sequences = np.zeros((len(sequences), MAX_LENGTH), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = seq[:MAX_LENGTH]
        padded_sequences[i, :len(seq)] = seq

    print('\n--- Example ---')
    idx_to_check = 5
    original_text = df[TEXT_COLUMN_NAME].iloc[idx_to_check]
    print(f'Original text: "{original_text}"')
    tokens = sp.encode_as_pieces(original_text)
    print(f'Tokens: {tokens}')
    padded_sequence = padded_sequences[idx_to_check]
    print(f'Padded sequence (length {len(padded_sequence)}):\n{padded_sequence}')
    
    np.save('processed_X_spm.npy', padded_sequences)
    np.save('processed_y_spm.npy', labels)
    print("\nData saved successfully.")


if __name__ == '__main__':
    main()