import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sentencepiece as spm
import json

def main():
    print('Loading data...')
    X = np.load('processed_X_spm.npy')
    y = np.load('processed_y_spm.npy')

    sp = spm.SentencePieceProcessor()
    sp.load('dota_chat_bpe.model')

    PARAMS_FILE = 'best_params_spm.json'
    print(f'Loading best hyperparams from "{PARAMS_FILE}"...')
    try:
        with open(PARAMS_FILE, 'r') as f:
            best_params = json.load(f)
        print(f'Using the best hyperparams: \n{best_params}')
    except FileNotFoundError:
        print(f'Error: File "{PARAMS_FILE}" with hyperparams was not found.')
        exit()

    VOCAB_SIZE = sp.GetPieceSize()
    NUM_CLASSES = len(np.unique(y))

    X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                stratify=y)
    train_val_data = TensorDataset(torch.from_numpy(X_train_val),
                                   torch.from_numpy(y_train_val).long())
    test_data = TensorDataset(torch.from_numpy(X_test),
                              torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_val_data,
                              batch_size=best_params['batch_size'],
                              shuffle=True)
    test_loader = DataLoader(test_data,
                             batch_size=best_params['batch_size'],
                             shuffle=False)
    
    final_model = SentimentLSTM(vocab_size=VOCAB_SIZE,
                                embedding_dim=best_params['embedding_dim'],
                                hidden_dim=best_params['hidden_dim'],
                                n_layers=best_params['n_layers'],
                                num_classes=NUM_CLASSES)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_class = getattr(optim, best_params['optimizer'])
    optimizer = optimizer_class(final_model.parameters(), lr=best_params['lr'])

    epochs = 15
    print(f'\nTraining the final model on {len(X_train_val)} examples over {epochs} epochs...')
    for epoch in range(epochs):
        final_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}] is complete.')
    print('Model was trained successfully!')

    final_model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = final_model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())

    final_accuracy = accuracy_score(y_test, all_preds)
    print(f'\nFinal accuracy on test: {final_accuracy:.2%} ')
    print('\nDetailed classification report on test data:')
    print(classification_report(y_test,
                                all_preds,
                                target_names=['Neutral',
                                              'Positive',
                                              'Negative']))

    FINAL_MODEL_PATH = 'dota_sentiment_lstm_spm.pth'
    torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
    print(f'\nThe final model is saved in "{FINAL_MODEL_PATH}"')

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

if __name__ == '__main__':
    main()