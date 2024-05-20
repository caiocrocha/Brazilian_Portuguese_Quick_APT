#!/usr/bin/env python3
# coding: utf-8

def main():
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "ɡ": 5, "v": 6, "ũ": 7, "õ": 8, "ʃ": 9, "m": 10, "i": 11, "p": 12, "w̃": 13, "f": 14, "ɛ": 15, "k": 16, "j̃": 17, "ɲ": 18, "s": 19, "a": 20, "l": 21, "ʒ": 22, "e": 23, "dʒ": 24, "ʎ": 25, "ã": 26, "n": 27, "b": 28, "ẽ": 29, "z": 30, "w": 31, "t": 32, "j": 33, "ĩ": 34, "tʃ": 35, "χ": 36, "ɔ": 37, "r": 38, "o": 39, "d": 40, "u": 41}
    print(f'Old vocab: {vocab}')

    chars = sorted(list(vocab.keys())[5:])

    sorted_vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}

    i = 5
    for c in chars:
        sorted_vocab[c] = i
        i+=1

    i = 1
    codes = {}
    for c in sorted_vocab:
        if len(c) == 2:
            codes[c] = str(i)
            i += 1
        else:
            codes[c] = c

    print(f'Vocab codes: {codes}')

    new_vocab = {}
    i = 0
    for v in codes.values():
        new_vocab[v] = i
        i += 1

    print(f'New vocab: {new_vocab}')

if __name__=='__main__': main()