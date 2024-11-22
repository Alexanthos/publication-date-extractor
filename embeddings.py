print('Import...')
import pandas as pd
import numpy as np
import fasttext
import fasttext.util
import requests

print('Load data...')
ft = fasttext.load_model('cc.fr.300.bin')
df = pd.read_csv('NLP_in_industry-original_data.csv')
df_date = pd.read_csv('gold_date.csv')
df = pd.merge(left=df, right=df_date, left_index=True, right_index=True)
df = df.drop(columns=['url', 'cache', 'nature', 'entity_type_datapolitics', 'entity_datapolitics'])

counter = 0
def read_url_txt(url):
    response = requests.get(url)
    text = response.text.split('\n')
    if '\n' in text:
        text.remove('\n')
    input_text = ['\n'.join(text[:50]), '\n'.join(text[-20:])]
    return '\n'.join(input_text)


def get_id_embed(id):
    return ft.get_word_vector(id)

def get_text_embeddings(url):
    global counter
    if counter % 10 == 0: print('Line', counter)

    text = read_url_txt(url)

    word_list = text.split()
    embeddings = np.zeros((len(word_list), 300))
    for idx, word in enumerate(word_list):
        embeddings[idx,:] = ft.get_word_vector(word)
    counter +=1

    return embeddings

print('compute embeddings...')
df['text_emb'] = df['text version'].map(get_text_embeddings)
df['id_embedding'] = df['doc_id'].map(get_id_embed)

print('save data file...')
df.to_pickle('data.pkl')