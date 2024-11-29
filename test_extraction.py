import pandas as pd

df = pd.read_pickle('df_with_regex_chunks.pkl')
print(f'# of NaN vals in the columns: {df["regex_chunks"].isna().sum()}')
print(df['regex_chunks'].head())
print(f'# of NaN vals in the columns: {df["regex_chunks_strict"].isna().sum()}')
print(df['regex_chunks_strict'].head())
print(df['regex_chunks'].map(len).value_counts())
print(df['regex_chunks_strict'].map(len).value_counts())
