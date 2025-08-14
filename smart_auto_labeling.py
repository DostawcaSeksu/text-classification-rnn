import pandas as pd
import re
from gensim.models import Word2Vec
import json

def main():
    MODEL_PATH = 'model\word2vec_dota_chat.model'
    KEYWORDS_PATH = 'data\seed_keywords.json'
    INPUT_FILE_PATH = 'data/chat.csv'
    OUTPUT_FILE_PATH = 'data/chat_labeled_smart.csv'
    TEXT_COLUMN_NAME = 'key'

    print(f'Loading model from {MODEL_PATH}...')
    try:
        model = Word2Vec.load(MODEL_PATH)
    except FileNotFoundError:
        print(f'Model {MODEL_PATH} not found.')
        exit()

    print(f'LOading keywords from "{KEYWORDS_PATH}"...')
    try:
        with open(KEYWORDS_PATH, 'r', encoding='UTF-8') as f:
            keywords = json.load(f)
        POSITIVE_SEED = keywords['positive']
        NEGATIVE_SEED = keywords['negative']
    except FileNotFoundError:
        print(f'Error. file {KEYWORDS_PATH} not found or have wrong format.') 
        exit()

    print('\nРасширение негативного словаря...')
    EXPANDED_NEGATIVE = expand_keywords(model, NEGATIVE_SEED, topn=10, threshold=0.4)
    print(f'Размер негативного словаря увеличен с {len(NEGATIVE_SEED)} до {len(EXPANDED_NEGATIVE)} слов.')

    print('\nРасширение позитивного словаря...')
    EXPANDED_POSITIVE = expand_keywords(model, POSITIVE_SEED, topn=10, threshold=0.45)
    print(f'Размер позитивного словаря увеличен с {len(POSITIVE_SEED)} до {len(EXPANDED_POSITIVE)} слов.')

def smart_label_sentiment(message, EXPANDED_NEGATIVE):
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
        if seed_word is model.wv:
            similar_words = model.wv.most_similar(seed_word, topn=topn)
            for word, score in similar_words:
                if score > threshold:
                    expanded_set.add(word)
    return expanded_set

if __name__ == '__main__':
    main()