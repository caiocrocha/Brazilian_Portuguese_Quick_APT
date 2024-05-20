#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
from datasets import load_metric
from argparse import ArgumentParser

def get_cmd_line():
    parser = ArgumentParser()
    parser.add_argument('-pred', required=True, type=str, 
                        help='Pred file containing the "transcript" col')
    parser.add_argument('-ref', required=True, type=str, 
                        help='Reference file containing the ref col')
    parser.add_argument('-ref_col', required=False, type=str, default='transcript_encoded', 
                        help='Col of the reference file containing the transcriptions')
    return parser.parse_args()

def main():
    args = get_cmd_line()
    pred = pd.read_csv(args.pred)
    ref = pd.read_csv(args.ref)
    
    cer_metric = load_metric("cer")

    reference = ref[args.ref_col].tolist()
    hypothesis = pred["transcript"].tolist()

    error = cer_metric.compute(predictions=hypothesis, references=reference)
    print(f'CER: {error}')

if __name__ == '__main__': main()