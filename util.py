# Authors: Girish Kumar Adari, Alexander Seljuk
# utility file

import pandas as pd


df = pd.read_csv('experiment_results copy.csv')

df['filter_combo'] = list(zip(df['num_filters1'], df['num_filters2']))

grouped = df.groupby('filter_combo')['accuracy'].mean().reset_index()

grouped['filter_combo'] = grouped['filter_combo'].apply(lambda x: f'{x[0]},{x[1]}')


df.to_csv('experiment_results2.csv', index=False)


