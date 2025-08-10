import pandas as pd
import re

def main():
    INPUT_FILE_PATH = 'data/chat.csv'
    OUTPUT_FILE_PATH = 'data/chat_labeled.csv'
    TEXT_COLUMN_NAME = 'key'

    print(f'Loading "{INPUT_FILE_PATH}" file')

    try:
        df = pd.read_csv(INPUT_FILE_PATH)
    except FileNotFoundError:
        print(f'Error: File "{INPUT_FILE_PATH}" was not found.')
        exit()

    df.dropna(subset=[TEXT_COLUMN_NAME], inplace=True)

    print('Starting auto data labeling...')

    df['sentiment'] = df[TEXT_COLUMN_NAME].apply(label_sentiment)

    print('Labeling was successfully complete.')
    print('\n--- Examples ---')
    print('\nPositive:')
    print(df[df['sentiment'] == 'Positive'][TEXT_COLUMN_NAME].sample(10).to_string(index=False))
    print('\nNegative:')
    print(df[df['sentiment'] == 'Negative'][TEXT_COLUMN_NAME].sample(10).to_string(index=False))
    print('\nNeutral:')
    print(df[df['sentiment'] == 'Neutral'][TEXT_COLUMN_NAME].sample(10).to_string(index=False))
    
    print(f'Saving labeled file: {OUTPUT_FILE_PATH}')
    df.to_csv(OUTPUT_FILE_PATH, index=False)
    print('Complete')

def label_sentiment(message):
    NEGATIVE_KEYWORDS = {
        'report', 'noob', 'idiot', 'end', 'ff', 'stupid', 'trash', 'cyka', 'blyat',
        'репорт', 'нуб', 'рак', 'дно', 'мусор', 'даун', 'afk', 'retard', 'faggot', 'shut up',
        'пидор', 'мразь', 'мать', 'ебал', 'сука', 'bitch', 'нахуй', 'fuck', 'fucked', 'family',
        'мама', 'отец', 'батя', 'сын', 'son', 'father', 'mother', 'fu'
    }

    POSITIVE_KEYWORDS = {
        'gg', 'wp', 'ggwp', 'gj', 'nice', 'ty', 'good', 'amazing', 'thanks', 'glhf', 'gl', 'hf',
        'молодец', 'спасибо', 'спс', 'красава', 'найс', 'гг', 'вп', 'ггвп', 'лайк', 'well played'
    }

    message = str(message).lower()
    clean_message = re.sub(r'[^\w\s]', '', message)

    words = set(clean_message.split())

    found_negative = words & NEGATIVE_KEYWORDS
    found_positive = words & POSITIVE_KEYWORDS

    if found_negative:
        return 'Negative'
    elif found_positive:
        return 'Positive'
    else:
        return 'Neutral'
    
if __name__ == '__main__':
    main()