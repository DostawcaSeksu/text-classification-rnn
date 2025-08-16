import pandas as pd
import sentencepiece as spm
import os

def main():
    return

def prepare_corpus(input_csv, text_column, output_txt):
    print(f'Loading data from "{input_csv}"...')
    try:
        df = pd.read_csv(input_csv)
        df.dropna(subset=[text_column], inplace=True)
        print(f'Saving the text corpus in {output_txt}')

        with open(output_txt, 'w', encoding='utf-8') as f:
            for text in df[text_column]:
                f.write(str(text) + '\n')

        return True
    except FileNotFoundError:
        return False
    except KeyError:
        return False
    
def train_spm_tokenizer(corpus_file, model_prefix, vocab_size, model_type='bpe'):
    print('\nStarting to train the SentencePiece tokenizer...')
    print(f'Vocab size: {vocab_size}, model type: {model_type}')

    command = (
        f'--input={corpus_file} --model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} --model_type={model_type} '
        f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
        f'--character_coverage=1.0'
    )

    spm.SentencePieceTrainer.Train(command)

    print('The tokenizer has been successfully trained')
    print(f'The model saved with "{model_prefix}" prefix (files .model and .vocab)')

def test_tokenizer(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')

    sentences_to_test = [
        'ggwp mid open',
        'report invoker noob',
        'репорт инвокера пидораса',
        'бездарь',
        'ьездарь',
        'бездврь'
    ]

    for sentence in sentences_to_test:
        pieces = sp.EncodeAsPieces(sentence)
        ids = sp.EncodeAsIds(sentence)
        print(f'\nOriginal: "{sentence}"')
        print(f'    -> Tokens: {pieces}')
        print(f'    -> Indexes: {ids}')

if __name__ == '__main__':
    main()