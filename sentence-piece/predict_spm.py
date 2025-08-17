import torch
import torch.nn as nn
import numpy as np
import json
import sentencepiece as spm

def main():
    print('Loading a tokenizator...')
    try:
        TOKENIZER_MODEL_PATH = 'dota_chat_bpe.model'
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_MODEL_PATH)
        VOCAB_SIZE = sp.GetPieceSize()
        print('Tokenizer was loaded successfully.')
    except FileNotFoundError:
        print(f'Error: "{TOKENIZER_MODEL_PATH}" was not found.')
        exit()

    PARAMS_FILE = 'best_params_spm.json'
    print(f'Loading best hyperparams from "{PARAMS_FILE}"...')
    try:
        with open(PARAMS_FILE, 'r') as f:
            best_params = json.load(f)
        print(f'Using the best hyperparams: \n{best_params}')
    except FileNotFoundError:
        print(f'Error: File "{PARAMS_FILE}" with hyperparams was not found.')
        exit()

    print('Loading the trained model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    NUM_CLASSES = 3
    model = SentimentLSTM(vocab_size=VOCAB_SIZE,
                          embedding_dim=best_params['embedding_dim'],
                          hidden_dim=best_params['hidden_dim'],
                          n_layers=best_params['n_layers'],
                          num_classes=NUM_CLASSES
    )

    MODEL_PATH = 'dota_sentiment_lstm_spm.pth'
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

        prediction = predict_sentiment(user_input, model, sp, device)
        print(f'{prediction}\n')

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, num_classes):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_hidden_state = hidden[-1]
        out = self.fc(final_hidden_state)
        return out

def predict_sentiment(text, model, sp_tokenizer, device, max_length=30):
    sequence = sp_tokenizer.EncodeAsIds(text)

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