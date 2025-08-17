# Dota 2 Chat Sentiment Analysis with PyTorch and LSTM

This project demonstrates a complete, modern MLOps pipeline for training a sentiment analysis model. The goal is to classify messages from Dota 2 in-game chat into three categories: **Positive**, **Negative**, or **Neutral**.

A key feature of this project is its advanced data preparation pipeline, which includes:
1.  **Semantic Weak Supervision:** Instead of simple keyword matching, it uses a pre-trained **Word2Vec** model to semantically expand a small seed dictionary, allowing for more nuanced and accurate automatic labeling.
2.  **Subword Tokenization:** It utilizes a **SentencePiece (BPE)** tokenizer trained from scratch on the Dota 2 chat corpus. This makes the model robust to out-of-vocabulary words, typos, and morphological variations common in noisy game chat.
3.  **Automated Hyperparameter Tuning:** It uses the **Optuna** framework to efficiently search for the optimal LSTM architecture and training configuration.

## Project Pipeline

The project is divided into a series of modular, executable Python scripts that represent a modern machine learning workflow.

### Module 1: Tooling (Executed Once)

These scripts build the core components needed for data processing.

1.  **`train_word2vec_gensim.py`** (Optional, from `01_word_embeddings` repo): Trains a `gensim.Word2Vec` model on the chat corpus. This model captures the semantic relationships between words and is used by the "smart" labeler.
2.  **`train_tokenizer_spm.py`**: Trains a **SentencePiece BPE tokenizer** on the raw chat data. This is the primary tool for converting text to numerical sequences.

### Module 2: Main Training Pipeline

1.  **`smart_auto_labeling.py`**: Reads the raw chat data and uses the trained Word2Vec model to semantically expand a small set of seed keywords. It then applies these expanded dictionaries to assign a sentiment label ('Positive', 'Negative', 'Neutral') to each message, creating `chat_labeled_smart.csv`.
2.  **`prepare_data_spm.py`**: Takes the labeled text data and uses the trained SentencePiece tokenizer to convert the text into padded numerical sequences. The processed data is saved in `.npy` format.
3.  **`tune_lstm.py`**: Uses the **Optuna** framework to perform an automated search for the best LSTM hyperparameters (layers, dimensions, learning rate, etc.). It trains dozens of models on a training set and evaluates them on a validation set. The best parameters are saved to `best_params_spm.json`.
4.  **`train_final_lstm.py`**: Loads the best hyperparameters, trains the final LSTM model on the combined training and validation data, and evaluates its performance on a held-out test set for a final, unbiased measure of its accuracy. The final model is saved to `dota_sentiment_lstm_spm.pth`.
5.  **`predict_spm.py`**: A simple inference script that loads the final trained model and tokenizer, allowing a user to input new phrases and get sentiment predictions in real-time.

## How to Run

### 1. Prerequisites
1. **Install Python 3.10.11**
2.  **Install CUDA toolkit 12.6 (you can dowload it [here](https://developer.nvidia.com/cuda-12-6-0-download-archive)).**
3. **An NVIDIA GPU with CUDA installed is highly recommended for training.**
4. **Create a virtual environment and install the required packages:**
  ```bash
  pip install -r req.txt
  ```

### 2. Prepare the Data & Tooling
Place your raw `chat.csv` file inside the `data/` directory.

```bash
# (Optional) First, train the Word2Vec model if you don't have one
# python ../01_word_embeddings/train_word2vec_gensim.py

# First, train the SentencePiece tokenizer
python train_tokenizer_spm.py

# Second, automatically label the raw chat data using the "smart" method
python smart_auto_labeling.py

# Third, convert the labeled text into numerical format using the SPM tokenizer
python prepare_data_spm.py
```
This will generate `chat_labeled_smart.csv`, `processed_X_spm.npy`, `processed_y_spm.npy`, and the tokenizer models.

### 3. Find the Best Hyperparameters
Run the Optuna optimization script. **This is a long process.**
```bash
python tune_lstm.py```
This will create a `best_params_spm.json` file.

### 4. Train and Evaluate the Final Model
Using the parameters found by Optuna, train the final model:
```bash
python train_final_model_spm.py
```
This script will train the best model and save its weights to `dota_sentiment_lstm_spm.pth`.

### 5. Run Inference
Interact with your trained model:
```bash
python predict_spm.py
```

## Results

### Hyperparameter Optimization
The Optuna search was run for 30 trials on the subword-tokenized data. The best performing model achieved a validation accuracy of **99.66%**.

*   **Best Hyperparameters:**
    ```json
    {
        "embedding_dim": 128,
        "hidden_dim": 256,
        "n_layers": 2,
        "lr": 0.0015479,
        "optimizer": "Adam",
        "batch_size": 128
    }
    ```

### Final Model Performance
The final model, trained with the optimal hyperparameters on the full training set, was evaluated on the held-out test set.

*   **Final Accuracy on Test Data: 99.76%**

*   **Classification Report:**
    ```
                  precision    recall  f1-score   support

         Neutral       1.00      1.00      1.00    164295
        Positive       1.00      1.00      1.00     46086
        Negative       1.00      1.00      1.00     77514

        accuracy                           1.00    287895
       macro avg       1.00      1.00      1.00    287895
    weighted avg       1.00      1.00      1.00    287895
    ```

The extremely high accuracy is a result of the model successfully learning the deterministic, semantically-expanded rules defined in the `smart_auto_labeling.py` script. The project demonstrates the power of a modern, end-to-end NLP pipeline.
