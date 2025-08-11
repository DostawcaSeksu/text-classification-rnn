import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from tune_pytorch import SentimentLSTM

print('Loading data and best hyperparameters...')
X = np.load('data/processed_X.npy')
y = np.load('data/processed_y.npy')

with open('word_to_idx.pickle', 'rb') as handle:
    word_to_idx = pickle.load(handle)

with open('best_params.json', 'r') as f:
    best_params = json.load(f)

print(best_params)

VOCAB_SIZE = len(word_to_idx)
NUM_CLASSES = len(np.unique(y))

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_val_data = TensorDataset(torch.from_numpy(X_train_val), torch.from_numpy(y_train_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_val_data, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=best_params['batch_size'], shuffle=False)

final_model = SentimentLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=best_params['embedding_dim'],
    hidden_dim=best_params['hidden_dim'],
    n_layers=best_params['n_layers'],
    num_classes=NUM_CLASSES
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
final_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_class = getattr(optim, best_params['optimizer'])
optimizer = optimizer_class(final_model.parameters(), lr=best_params['lr'])

epochs = 15
print(f'Train final model on {len(X_train_val)} examples over {epochs} epochs...')
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

final_model.eval()
all_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = final_model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        all_preds.extend(predicted.cpu().numpy())

final_accuracy = accuracy_score(y_test, all_preds)
print(f'\nFinal accuracy on test: {final_accuracy:.2%} ')
print('\nDetailed classification report on test data:')
print(classification_report(y_test, all_preds, target_names=['Neutral', 'Positive', 'Negative']))

FINAL_MODEL_PATH = 'dota_sentiment_lstm.pth'
torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
print(f'\nThe final model is saved in "{FINAL_MODEL_PATH}"')