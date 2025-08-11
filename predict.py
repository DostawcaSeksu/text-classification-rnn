import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import re
from nltk.tokenize import word_tokenize
from tune_pytorch import SentimentLSTM

def main():
    print('Loading a vocab...')
    DATA_PATH = 'data/'
    with open(DATA_PATH + 'word_to_idx.pickle', 'rb') as handle:
        word_to_idx = pickle.load(handle)
    VOCAB_SIZE = len(word_to_idx)
    print('Vocab loaded successfully')

    print('Loading a hyperparameters...')
    with open(DATA_PATH + 'best_params.json', 'r') as f:
        best_params = json.load(f)
    NUM_CLASSES = 3
    print('Hyperparameters loaded successfully.')

    print('Loading the trained model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentimentLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=best_params['embedding_dim'],
        hidden_dim=best_params['hidden_dim'],
        n_layers=best_params['n_layers'],
        num_classes=NUM_CLASSES
    )

    MODEL_PATH = 'dota_sentiment_lstm.pth'
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    print(f'Model "{MODEL_PATH}" loaded on "{device}".')

    print('\nTesting model')
    print('\nType random dota2 prase(or "exit" to exit): ')

    while True:
        user_input = input('> ')
        if user_input.lower() == 'exit':
            break

        prediction = predict_sentiment(user_input, model, word_to_idx, device)
        print(f'{prediction}\n')

def predict_sentiment(text, model, word_to_idx, device, max_length=30):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
    tokens = word_tokenize(text)

    sequence = [word_to_idx.get(word, 1) for word in tokens]

    padded_sequence = np.zeros(max_length, dtype=np.int64)
    sequence = sequence[:max_length]
    padded_sequence[:len(sequence)] = sequence

    input_tensor = torch.tensor(padded_sequence).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        
    idx_to_label = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
    
    return idx_to_label.get(predicted_idx, "Unknown")

if __name__ == '__main__':
    main()