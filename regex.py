import re
import pandas as pd

def load_from_csv(path_original_data, path_gold):
    df = pd.read_csv(path_original_data)
    df_date = pd.read_csv(path_gold)
    df = pd.merge(left=df, right=df_date, left_index=True, right_index=True)
    df = df.drop(columns=['url', 'cache', 'nature', 'entity_type_datapolitics', 'entity_datapolitics'])
    return df

def regex_on_df(regex, row):
    splitting_regex = r"\r|\n"
    sentences = re.split(splitting_regex, row)
    # sentences = row.split('\n')
    matching_lines = [sent for sent in sentences if re.search(regex, sent)]
    return matching_lines

if __name__=='__main__':
    df = load_from_csv('NLP_in_industry-original_data.csv', 'gold_date.csv')
    # filtered_df = df.dropna(subset=["Text"])
    # print(df.info())
    regex_for_data_extraction = r"\b(\d{1,2})\s*[./-]?\s*(?:([a-zéû]{3,10})|(\d{1,2}))\s*[./-]?\s*(\d{4})?\b"
    df['dates'] = df['Text'].apply(lambda x: regex_on_df(regex_for_data_extraction, str(x)))
    print(df['dates'])