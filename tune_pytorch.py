import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
from functools import partial
import logging
import sys
import json

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter('%(message)s'))
    
    file_handler = logging.FileHandler('tuning.log', mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger

def main():
    logger = setup_logging()
    
    DATA_PATH = 'data/'
    logger.info('Loading preprocessed data...')
    X = np.load(DATA_PATH + 'processed_X.npy')
    y = np.load(DATA_PATH + 'processed_y.npy')
    
    with open('word_to_idx.pickle', 'rb') as handle:
        word_to_idx = pickle.load(handle)

    VOCAB_SIZE = len(word_to_idx)
    NUM_CLASSES = len(np.unique(y))    
    logger.info(f'Vocab size: {VOCAB_SIZE}, Number of classes: {NUM_CLASSES}')

    X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
    
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    logger.info('\nStarting to Find the Best Hyperparameters with Optuna...')
    
    objective_with_data = partial(objective, train_dataset=train_data, val_dataset=val_data, vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
    
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_data, n_trials=30)
    
    logger.info('\n--- Results of Optuna search ---')
    best_trial = study.best_trial
    logger.info(f'Best accuracy: {best_trial.value:.2%}')
    logger.info('Best hyperparams: ')
    for key, value in best_trial.params.items():
        logger.info(f'    {key}: {value}')
        
    BEST_PARAMS_FILE = 'best_params.json'
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
    logger.info(f"\nBest hyperparams was saved in '{BEST_PARAMS_FILE}'")

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
    
def objective(trial, train_dataset, val_dataset, vocab_size, num_classes):
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    n_layers = trial.suggest_int('n_layers', 1, 2)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SentimentLSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

if __name__ == '__main__':
    main()