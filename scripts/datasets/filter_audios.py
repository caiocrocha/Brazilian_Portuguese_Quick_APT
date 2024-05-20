#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Filter audios to compose the train set according to the following selected criteria:
1 - Variety = PT-BR
2 - Up votes > 0 and down votes == 0
3 - Length(transcript) > 2
4 - Q1(Audio duration distribution) < Audio duration < Q3(Audio duration distribution)

Command line arguments:
1 - CORAA metadata file
2 - File containing the audio durations for each file_path in the CORAA metadata file
3 - Output file containing the audio IDs
"""

import pandas as pd
import sys

def main():
    df = pd.read_csv(sys.argv[1]).set_index('file_path')
    print(f'Initial shape: {df.shape}')
    df = df.query('variety == "pt_br"')
    print(f'Shape after removing pt_pt audios: {df.shape}')
    df = df.query('up_votes > 0 and down_votes == 0')
    print(f'Shape after applying rules to filter out transcriptions of bad quality: {df.shape}')
    short_transcript = df['transcript_ipa'].apply(lambda v: len(v) < 3)
    df = df.loc[~short_transcript]
    print(f'Shape after removing audios with short transcriptions: {df.shape}')

    df_durs = pd.read_csv(sys.argv[2]).set_index('file_path')
    df_merged = pd.merge(df, df_durs, left_index=True, right_index=True)
    metrics = df_merged['duration'].describe()
    df_merged = df_merged.query(f"{metrics.loc['25%']} < duration < {metrics.loc['75%']}")
    print(f"Shape after filtering by Q1 = {metrics.loc['25%']} and Q3 = {metrics.loc['75%']}: {df_merged.shape}")
    dataset_size = df_merged['duration'].sum() / 3600
    print(f"Total dataset size: {dataset_size:.2f}h")
    # df_merged = df_merged.sample(frac=10/dataset_size, random_state=42)
    sample_dataset_size = df_merged['duration'].sum() / 3600
    print(f"Sample dataset size: {sample_dataset_size:.2f} min")
    df = df.loc[df_merged.index]
    # df.reset_index()['file_path'].to_csv(sys.argv[2], index=False, header=False)
    df.to_csv(sys.argv[3])
    print(f'Audio IDs saved to {sys.argv[2]}')

if __name__ == '__main__': main()