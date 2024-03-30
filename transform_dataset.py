#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Transform the X-SAMPA transcriptions of the dataframe of transcriptions to IPA

Arguments:
    Input CSV file: The CSV file of X-SAMPA phonetic transcriptions
    Output CSV filename: The output file of X-SAMPA and IPA phonetic transcriptions
"""

import sys
import pandas as pd

xsampa2ipa = {
    'l': 'l',
    'u': 'u',
    'k': 'k',
    'a': 'a',
    's': 's',
    'S': 'ʃ',
    'r': 'r',
    't': 't',
    'm': 'm',
    'i': 'i',
    'o': 'o',
    'f': 'f',
    'w': 'w',
    'Z': 'ʒ',
    'e~': 'ẽ',
    'j~': 'j̃',
    'R': 'ʁ',
    'e': 'e',
    'j': 'j',
    'i~': 'ĩ',
    'n': 'n',
    'z': 'z',
    'v': 'v',
    'a~': 'ã',
    'w~': 'w̃',
    'E': 'ɛ',
    'b': 'b',
    'X': 'χ',
    'd': 'd',
    'dZ': 'dʒ',
    'p': 'p',
    'O': 'ɔ',
    'g': 'ɡ',
    'o~': 'õ',
    'tS': 'tʃ',
    'u~': 'ũ',
    'J': 'ɲ',
    'L': 'ʎ'
}

def translate_xsampa(text):
    """
    Transforma XSAMPA para IPA (saída do falabrasil é em XSAMPA)
    """
    return ' '.join(''.join(xsampa2ipa[c] for c in word.split(' ')) for word in text.split('  '))

def main():
    if len(sys.argv) < 3:
        print('Missing arguments:\nInput CSV file\nOutput CSV filepath')
        return

    path = sys.argv[1]
    output = sys.argv[2]
    df = pd.read_csv(path)
    df['g2p_ipa'] = df['g2p'].apply(translate_xsampa)
    df.to_csv(output, index=False)

if __name__ == '__main__':
    main()