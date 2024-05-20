#!/usr/bin/env python
# coding: utf-8

codes = {'<pad>': '<pad>', '<s>': '<s>', '</s>': '</s>', '<unk>': '<unk>', 
         '|': '|', 'a': 'a', 'b': 'b', 'd': 'd', 'dʒ': '1', 'e': 'e', 'f': 'f', 
         'i': 'i', 'j': 'j', 'j̃': '2', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 
         'o': 'o', 'p': 'p', 'r': 'r', 's': 's', 't': 't', 'tʃ': '3', 'u': 'u', 
         'v': 'v', 'w': 'w', 'w̃': '4', 'z': 'z', 'ã': 'ã', 'õ': 'õ', 'ĩ': 'ĩ', 
         'ũ': 'ũ', 'ɔ': 'ɔ', 'ɛ': 'ɛ', 'ɡ': 'ɡ', 'ɲ': 'ɲ', 'ʃ': 'ʃ', 'ʎ': 'ʎ', 
         'ʒ': 'ʒ', 'χ': 'χ', 'ẽ': 'ẽ'}

symbol2phoneme = {v: k for k, v in codes.items()}

def decode(transcript):
    return ''.join(symbol2phoneme[k] if k in {'a', 'b', 'd', '1', 'e', 'f', 'i', 'j', '2', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', '3', 'u', 'v', 'w', '4', 'z', 'ã', 'õ', 'ĩ', 'ũ', 'ɔ', 'ɛ', 'ɡ', 'ɲ', 'ʃ', 'ʎ', 'ʒ', 'χ', 'ẽ'} else k for k in transcript)
