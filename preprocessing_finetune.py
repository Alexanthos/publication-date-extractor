import requests
from datasets import load_dataset, Dataset
import pandas as pd

def get_text_doc(link, length = 4_000):
    print(f'Fetching link {link}')
    def fetch_text(link):
        try:
            response = requests.get(link)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {link}: {e}")
            return None

    text = fetch_text(link)
    if text is None: return None
    text_start = text[:length]
    text_end = text[-length:]
    text = text_start + " " + text_end
    return text


def get_df_text_format(org_df_path, length):
    """
    Input:
        org_df_path: str, a path to a dataframe containing at least 2 colunns : url (urls to match the rows on) and 'text version' (the links to text versions of the documents)
        length: int, the number of characters to take from the beginning and end of the document (length=3000 will result in a document of max length = 6000)
    Output:
        a pd.DataFrame object with a column 'text' that contains the text of the documents + all the original columns of that dataframe
    """
    dataset = load_dataset("maribr/publication_dates_fr")['train']
    df_hf = dataset.to_pandas()
    df = pd.read_csv(org_df_path)[['url','text version']]
    df_hf = df_hf.merge(df, on='url').drop(columns='Text')
    df_hf['text'] = df_hf['text version'].apply(lambda x : get_text_doc(x, length=length))
    return df_hf

def get_dataset_text_format(org_df_path, length, dropna=True):
    """
    Input:
        org_df_path: str, a path to a dataframe containing at least 2 colunns : url (urls to match the rows on) and 'text version' (the links to text versions of the documents)
        length: int, the number of characters to take from the beginning and end of the document (length=3000 will result in a document of max length = 6000)
    Output:
        a DatasetDict object with a train - test split
    """
    df_hf = get_df_text_format(org_df_path, length)
    # dropping the rows where fetching the text from url failed
    if dropna:
        df_hf = df_hf.dropna()
    # creating a train-test split
    dataset = Dataset.from_pandas(df_hf)
    dataset_train_test = dataset.train_test_split(test_size = 0.3, seed=42)
    return dataset_train_test
