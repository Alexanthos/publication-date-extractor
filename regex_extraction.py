import re
import pandas as pd
import requests
from datasets import Dataset
from datasets import load_dataset

def load_from_csv(path_original_data, path_gold):
    df = pd.read_csv(path_original_data)
    df_date = pd.read_csv(path_gold)
    df = pd.merge(left=df, right=df_date, left_index=True, right_index=True)
    df = df.drop(columns=['nature', 'entity_type_datapolitics', 'entity_datapolitics'])
    return df

def load_datasets(path_hf, path_original_data):
    df_original = pd.read_csv(path_original_data)
    ds = load_dataset(path_hf)
    df_hf = ds['train'].to_pandas()
    df = pd.merge(left=df_original, right=df_hf, on='url')
    return df

def regex_on_df(regex, row):
    splitting_regex = r"\r|\n|\."
    text = row[:4000] if row else ''
    sentences = re.split(splitting_regex, text)
    # sentences = row.split('\n')
    matching_lines = [sent for sent in sentences if re.search(regex, sent)]
    return matching_lines if matching_lines else ''


def add_chunks_from_links(link, regex_for_data_extraction):
    def fetch_text(link):
        try:
            response = requests.get(link)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {link}: {e}")
            return None

    text = fetch_text(link)
    chunks = regex_on_df(regex_for_data_extraction, text)
    return chunks



if __name__=='__main__':
    regex_for_data_extraction = r"\b(\d{1,2})\s*(?:([a-zéûA-ZÉÛ]{3,10})|([/-]?\s*\d{1,2}))\s*[./-]?\s*(\d{4})?\b"
    regex_for_data_extraction_strict = r"\b\d{1,2}[./-]?\s*(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|JANVIER|FÉVRIER|MARS|AVRIL|MAI|JUIN|JUILLET|AOÛT|SEPTEMBRE|OCTOBRE|NOVEMBRE|DÉCEMBRE)\s*[./-]?\s*\d{4}\b"
    df = load_datasets("maribr/publication_dates_fr", "NLP_in_industry-original_data.csv")
    df['regex_chunks'] = df['text version'].apply(lambda x: add_chunks_from_links(x,regex_for_data_extraction))
    df['regex_chunks_strict'] = df['text version'].apply(lambda x: add_chunks_from_links(x,regex_for_data_extraction_strict))
    df.to_pickle('df_with_regex_chunks.pkl')