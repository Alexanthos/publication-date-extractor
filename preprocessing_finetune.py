import requests
from datasets import load_dataset, Dataset
import pandas as pd
import argparse
import os


def get_text_doc(link, length = 3_000):
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

def get_dataset_text_format(org_df_path, length, dropna=True, split = True):
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
    if split:
        dataset = dataset.train_test_split(test_size = 0.3, seed=42)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for finetuning or finetuned Llama inference.")
    parser.add_argument("--input-path", type=str, help="Path to file that you want to label. Can be .csv or .pkl")
    parser.add_argument("--out-type", type=str, default = 'csv', help="Output path and filename. Can be 'csv', 'pkl' (pandas Dataframe) or 'dataset' (HuggingFace DatasetDict)")
    parser.add_argument("--output-path", type=str, help="The path where the output file will be saved")
    parser.add_argument("--length", type=int, default= 3_000, help="The length of the start and end chunks, in characters")
    parser.add_argument("--no-split", action='store_true', help = "When the script is run with this flag don't split the dataset into train and test")
    
    
    args = parser.parse_args()
    
    assert args.out_type in ('csv', 'pkl', 'dataset'), f"Incorrect --out-type. Has to be one of ('csv', 'pkl', 'dataset'), not {args.out_type}"
    
    
    dataset = get_dataset_text_format(args.input_path, args.length, split= not args.no_split)
    
    print(dataset)
    
    print(f'Saving the dataset to {args.output_path}')
    
    if args.out_type == 'dataset':
        dataset.save_to_disk(args.output_path)
    else:
        if not args.no_split:
            data_train, data_test = dataset['train'], dataset['test']
            # making the directory where the dataframes will be saved
            os.makedirs(args.output_path, exist_ok=True)
            df_train, df_test = data_train.to_pandas(), data_test.to_pandas()
            
            train_path = os.path.join(args.output_path,f'df_train.{args.out_type}')
            test_path = os.path.join(args.output_path,f'df_test.{args.out_type}')
            
            if args.out_type == 'csv':
                df_train.to_csv(train_path)
                df_test.to_csv(test_path)
            elif args.out_type == 'pkl':
                df_train.to_pickle(train_path)
                df_test.to_pickle(test_path)
        else:
            df = dataset.to_pandas()
            if args.out_type == 'csv':
                df.to_csv(f"{args.output_path}.{args.out_type}")
            elif args.out_type == 'pkl':
                df.to_pickle(f"{args.output_path}.{args.out_type}")

if __name__=='__main__':
    main()