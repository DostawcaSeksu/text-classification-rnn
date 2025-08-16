import pandas as pd
import re
from gensim.models import Word2Vec
import json
import os

def main():
    MODEL_PATH = os.path.join('../w2v-model', 'word2vec_dota_chat.model')
    DATA_FOLDER = '../data/'
    KEYWORDS_PATH = os.path.join(DATA_FOLDER, 'seed_keywords.json')
    INPUT_FILE_PATH = os.path.join(DATA_FOLDER, 'chat.csv')
    OUTPUT_FILE_PATH = 'chat_labeled_smart.csv'
    TEXT_COLUMN_NAME = 'key'

    print(f'Loading model from {MODEL_PATH}...')
    try:
        model = Word2Vec.load(MODEL_PATH)
    except FileNotFoundError:
        print(f'Model {MODEL_PATH} not found.')
        exit()

    print(f'Loading keywords from "{KEYWORDS_PATH}"...')
    try:
        with open(KEYWORDS_PATH, 'r', encoding='UTF-8') as f:
            keywords = json.load(f)
        POSITIVE_SEED = keywords['positive']
        NEGATIVE_SEED = keywords['negative']
    except (FileNotFoundError, json.JSONDecodeError):
        print(f'Error: file {KEYWORDS_PATH} not found or has wrong format.') 
        exit()

    print('\nExpanding the Negative Vocabulary...')
    EXPANDED_NEGATIVE = expand_keywords(model, NEGATIVE_SEED, topn=10, threshold=0.4)
    print(f'The size of the negative dictionary has been increased from {len(NEGATIVE_SEED)} to {len(EXPANDED_NEGATIVE)} words.')

    print('\nExpanding the Positive Vocabulary...')
    EXPANDED_POSITIVE = expand_keywords(model, POSITIVE_SEED, topn=10, threshold=0.45)
    print(f'The size of the positive dictionary has been increased from {len(POSITIVE_SEED)} to {len(EXPANDED_POSITIVE)} words.')

    print(f'\nUploading and labeling "{INPUT_FILE_PATH}"...')
    df = pd.read_csv(INPUT_FILE_PATH)
    df.dropna(subset=[TEXT_COLUMN_NAME], inplace=True)
    df['sentiment'] = df[TEXT_COLUMN_NAME].apply(
        lambda msg: smart_label_sentiment(msg, EXPANDED_NEGATIVE, EXPANDED_POSITIVE))

    print('\n---Statistics on the new markup---')
    print(df['sentiment'].value_counts())
    
    df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f'\nSmartly tagged data is stored in "{OUTPUT_FILE_PATH}"')

def smart_label_sentiment(message, EXPANDED_NEGATIVE, EXPANDED_POSITIVE):
    message = str(message).lower()
    clean_message = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', message)
    words = set(clean_message.split())

    if words & EXPANDED_NEGATIVE:
        return 'Negative'
    if words & EXPANDED_POSITIVE:
        return 'Positive'
    return 'Neutral'
    
def expand_keywords(model, seed_keywords, topn=10, threshold=0.5):
    seed_set = set(seed_keywords)
    expanded_set = set(seed_keywords)

    for seed_word in seed_set:
        if seed_word in model.wv:
            similar_words = model.wv.most_similar(seed_word, topn=topn)
            for word, score in similar_words:
                if score > threshold:
                    expanded_set.add(word)
    return expanded_set

if __name__ == '__main__':
    main()