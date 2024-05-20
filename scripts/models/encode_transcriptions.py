#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
from PString import PString
import sys

codes = {'<pad>': '<pad>', '<s>': '<s>', '</s>': '</s>', '<unk>': '<unk>', 
         '|': '|', 'a': 'a', 'b': 'b', 'd': 'd', 'dʒ': '1', 'e': 'e', 'f': 'f', 
         'i': 'i', 'j': 'j', 'j̃': '2', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 
         'o': 'o', 'p': 'p', 'r': 'r', 's': 's', 't': 't', 'tʃ': '3', 'u': 'u', 
         'v': 'v', 'w': 'w', 'w̃': '4', 'z': 'z', 'ã': 'ã', 'õ': 'õ', 'ĩ': 'ĩ', 
         'ũ': 'ũ', 'ɔ': 'ɔ', 'ɛ': 'ɛ', 'ɡ': 'ɡ', 'ɲ': 'ɲ', 'ʃ': 'ʃ', 'ʎ': 'ʎ', 
         'ʒ': 'ʒ', 'χ': 'χ', 'ẽ': 'ẽ'}

def main():
    df = pd.read_csv(sys.argv[1])
    df['transcript_encoded'] = df['transcript_ipa'].apply(
        lambda transcript: ''.join(codes[k] if k != ' ' else ' ' for k in PString(transcript)))
    df.to_csv(sys.argv[2], index=False)

if __name__ == '__main__': main()