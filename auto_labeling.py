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
    'noob', 'n00b', 'newb', 'newbie', 'idiot', 'moron', 'stupid', 'dumb', 'dumbass', 'retard',
    'trash', 'garbage', 'shit', 'sh1t', 'crap', 'rubbish', 'loser', 'faggot', 'fag', 'gay',
    'bitch', 'btch', 'whore', 'slut', 'asshole', 'ass', 'a55', 'douche', 'douchebag',
    'fuck', 'fck', 'fuk', 'fucked', 'fucker', 'motherfucker', 'mf', 'shithead', 'dipshit',
    'bastard', 'prick', 'dick', 'd1ck', 'cock', 'pussy', 'pu55y', 'twat', 'wanker',
    'scrub', 'feeder', 'troll', 'troller', 'griefer', 'thrower', 'afk', 'afker', 'leaver',
    'quit', 'report', 'rpt', 'rep', 'end', 'ff', 'suck', 'sucks', 'bad', 'terrible',
    'awful', 'horrible', 'useless', 'worthless', 'dog', 'dogshit', 'ez', 'easy', 'noskill',
    'braindead', 'brainless', 'r3tard', 'rtard', 'cancer', 'toxic', 'shutup', 'stfu', 'gtfo',
    'kys', 'die', 'uninstall', 'lowskill', 'boosted', 'smurf', 'smurfer', 'ruin', 'ruiner',
    'нуб', 'нубас', 'новичок', 'рак', 'раковый', 'дно', 'даун', 'дебил', 'идиот', 'тупой',
    'глупый', 'тормоз', 'мудак', 'мусор', 'дерьмо', 'говно', 'ублюдок', 'чмо', 'чушок',
    'гандон', 'засранец', 'долбаеб', 'долбоеб', 'бездарь', 'бездарный', 'нищий', 'бомж',
    'лох', 'лошара', 'отсталый', 'тупая', 'тупое', 'кретин', 'урод', 'дебилоид', 'мразь',
    'тварь', 'падла', 'падло', 'сволочь', 'придурок', 'дурак', 'дурачок', 'тупица',
    'олень', 'осел', 'баран', 'козел', 'дебелый', 'тупорылый', 'тупоголовый', 'туполобый',
    'дебильный', 'идиотский', 'рачина', 'рачище', 'шваль', 'отребье', 'позорник', 'позорный',
    'хуй', 'хуе', 'хуйло', 'хуйня', 'хуета', 'похер', 'пох', 'похуй', 'нахуй', 'нахер',
    'ебать', 'ебал', 'еби', 'ебаный', 'еблан', 'еба', 'ебарь', 'ебло', 'ебан', 'ебнутый',
    'ебанутый', 'сука', 'сучка', 'пидор', 'пидорас', 'пидр', 'похер', 'хуесос', 'хуйсос',
    'блядь', 'бля', 'блять', 'бляха', 'еблище', 'ебучка', 'похерист', 'хуяк', 'хуйпох',
    'похуист', 'ебизм', 'ебанатик', 'ебанашка', 'хуйлуша', 'похаб', 'похабный', 'хуевый',
    'хуёво', 'хуйовый', 'ебливый', 'ебанство', 'хуйство', 'пидорство', 'похуизм',
    'репорт', 'рпт', 'реп', 'слив', 'сливщик', 'фидер', 'корм', 'афк', 'афка', 'ливер',
    'свалил', 'конец', 'фф', 'похеризм', 'раки', 'нубы', 'дауны', 'фидера', 'ливеры',
    'афкеры', 'афкашник', 'слива', 'кормило', 'кормильщик', 'токсик', 'токсичный',
    'мамка', 'мама', 'мать', 'батя', 'отец', 'сын', 'дочь', 'мамку', 'матери', 'папаша',
    'сынок', 'дочурка', 'матюгальник',
    'cyka', 'suka', 'blyat', 'blya', 'blyad', 'pidor', 'pidaras', 'pidr', 'mudak', 'debil',
    'rak', 'dno', 'chmo', 'gandon', 'ebal', 'ebat', 'eban', 'ebanutiy', 'naxuy', 'naxer',
    'poxer', 'poxuy', 'xuy', 'xuj', 'xui', 'xyi', 'xyilo', 'xuylo', 'sosi', 'sosy',
    'lowbob', 'lowbobik', 'noobik', 'tupitsa', 'durak', 'durachok', 'xuesos', 'xuylo',
    'eblan', 'eblishe', 'suka', 'blyadina', 'poh', 'pohab', 'xuyak', 'ebaka', 'ebanashka'
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