#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns

def main():
    df = pd.read_csv(sys.argv[1])
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    sns.boxplot(df['duration'], ax=ax[0])
    sns.boxplot(df['duration'], showfliers=False, ax=ax[1])
    ax[0].set_ylabel('Duration (s)')
    ax[0].set_xlabel('With outliers')
    ax[1].set_xlabel('Without outliers')
    ax[0].set(xticks=[], xticklabels=[])
    ax[1].set(xticks=[], xticklabels=[])
    plt.savefig('boxplot_duration.pdf')
    plt.show()

if __name__ == '__main__': main()