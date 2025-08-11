
# Dota 2 Chat Sentiment Analysis with LSTM

This project demonstrates a complete MLOps pipeline for training a sentiment analysis model using PyTorch and LSTMs. The goal is to classify messages from Dota 2 in-game chat into three categories: **Positive**, **Negative**, or **Neutral**.

A key feature of this project is the use of **weak supervision** to automatically label a large, unlabeled chat dataset, followed by hyperparameter optimization with **Optuna** to find the best possible model architecture.

## Project Pipeline

The project is divided into a series of modular, executable Python scripts that represent a standard machine learning workflow:

1.  **`auto_labeling.py`**: Reads the raw chat data (`chat.csv`) and applies a set of heuristic rules based on keyword dictionaries to assign a sentiment label ('Positive', 'Negative', 'Neutral') to each message. The output is a new labeled CSV file (`chat_labeled.csv`).
2.  **`prepare_data.py`**: Takes the labeled text data, builds a vocabulary, and converts the text into padded numerical sequences suitable for training a neural network. The processed data is saved in `.npy` format for efficient loading.
3.  **`tune_pytorch.py`**: Uses the **Optuna** framework to perform an automated search for the best model hyperparameters. It trains dozens of different LSTM models on a training set and evaluates them on a validation set to find the optimal architecture and training configuration. The best parameters are saved to `best_params.json`.
4.  **`train.py`**: Loads the best hyperparameters found by Optuna, trains the final LSTM model on the combined training and validation data, and evaluates its performance on a held-out test set to get a final, unbiased measure of its accuracy.

## How to Run

### 1. Prerequisites

1. **Install Python 3.10.11**
2.  **Install CUDA toolkit 12.6 (you can dowload it [here](https://developer.nvidia.com/cuda-12-6-0-download-archive)).**
3. **An NVIDIA GPU with CUDA installed is highly recommended for training.**
4. **Create a virtual environment and install the required packages:**
  ```bash
  pip install -r req.txt
  ```

### 2. Prepare the Data

Place your raw `chat.csv` file inside the `data/` directory. Then, run the data preparation pipeline:

```bash
# First, automatically label the raw chat data
python auto_labeling.py

# Second, convert the labeled text into numerical format
python prepare_data.py
```
This will generate `chat_labeled.csv`, `processed_X.npy`, `processed_y.npy`, and `word_to_idx.pickle`.

### 3. Find the Best Hyperparameters

Run the Optuna optimization script. **This is a long process and may take several hours.**

```bash
python tune_pytorch.py
```
This will create a `best_params.json` file containing the optimal model configuration.

### 4. Train the Final Model

Using the parameters found by Optuna, train the final model:

```bash
python train_final_model.py
```
This script will train the best model and save its weights to `dota_sentiment_lstm.pth`.

## Results

### Hyperparameter Optimization

The Optuna search was run for 30 trials. The best performing model was found with the following parameters, achieving a validation accuracy of **99.96%**.

```
--- Results of Optuna search ---
Best shot:
  accuracy: 99.96%
  Best hyperparams:
    embedding_dim: 256
    hidden_dim: 256
    n_layers: 1
    lr: 0.0011198390085146272
    optimizer: RMSprop
    batch_size: 128
```

### Final Model Performance

The final model, trained with the optimal hyperparameters, was evaluated on the held-out test set.

**Final accuracy on test data: 99.95%**

**Detailed classification report:**
```
              precision    recall  f1-score   support

     Neutral       1.00      1.00      1.00    226774
    Positive       1.00      1.00      1.00     40411
    Negative       1.00      1.00      1.00     20710

    accuracy                           1.00    287895
   macro avg       1.00      1.00      1.00    287895
weighted avg       1.00      1.00      1.00    287895
```

The extremely high accuracy is a result of the model successfully learning the deterministic rules defined in the `auto_labeling.py` script.