#/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import yaml
import json
import torch
import jiwer
import numpy as np
import jiwer.transforms as tr
import random
import pandas as pd

import torchaudio
import librosa

from shutil import copyfile
# from utils.generic_utils import load_config, load_vocab

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from datasets import load_dataset, load_metric, concatenate_datasets
from datasets import ClassLabel

import transformers
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers import EarlyStoppingCallback

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=True,
                    help="json file with configurations")
# parser.add_argument('--checkpoint_path', type=str, default='facebook/wav2vec2-xls-r-1b',
parser.add_argument('--checkpoint_path', type=str, default='facebook/wav2vec2-large-xlsr-53',
                    help="path of checkpoint pt file, for continue training")
parser.add_argument('--continue_train',
                    default=False,
                    action='store_true',
                    help='If True Continue the training using the checkpoint_path')

args = parser.parse_args()
print(f'Using checkpoint {args.checkpoint_path}!')
if args.continue_train:
    print('Continue training from last checkpoint')

class SentencesToListOfCharacters(tr.AbstractTransform):
    def process_string(self, s):
        return list(s)

    def process_list(self, inp):
        chars = []
        for sentence in inp:
            chars.extend(self.process_string(sentence))

        return chars

# cer_transform = tr.Compose(
#     [
#         jiwer.RemoveMultipleSpaces(),
#         jiwer.Strip(),
#         SentencesToListOfCharacters(), # convert words to chars
#         # jiwer.RemoveEmptyStrings()  # remove space strings
#     ]
# )

# It's the jiwer default transform
# wer_transform = jiwer.Compose([
#     jiwer.RemoveMultipleSpaces(),
#     jiwer.Strip(),
#     jiwer.SentencesToListOfWords(),
#     jiwer.RemoveEmptyStrings()
# ])

def compute_cer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    # cer = jiwer.wer(reference, hypothesis, truth_transform=cer_transform, hypothesis_transform=cer_transform)
    cer = jiwer.cer(reference, hypothesis)
    return cer

def compute_wer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    # wer = jiwer.wer(reference, hypothesis, truth_transform=wer_transform, hypothesis_transform=wer_transform)
    wer = jiwer.wer(reference, hypothesis)
    return wer

def replace_special_tokens_and_normalize(text, vocab_string, processor):
    text = text.lower()
    text = text.replace(processor.tokenizer.unk_token, " ")
    text = text.replace(processor.tokenizer.pad_token, " ")
    text = text.replace(processor.tokenizer.word_delimiter_token, " ")
    text = re.sub("[^{}]".format(vocab_string+" "), " ", text)
    text = re.sub("[ ]+", " ", text)
    # remove doble blank spaces
    text = " ".join(text.split())
    return text

def calculate_wer(pred_ids, labels, processor, vocab_string, debug=False):
    labels[labels == -100] = processor.tokenizer.pad_token_id

    pred_string = processor.batch_decode(pred_ids)
    label_string = processor.batch_decode(labels, group_tokens=False)
    # wer = wer_metric.compute(predictions=pred_string, references=label_string)
    wer = 0
    cer = 0
    for i in range(len(pred_string)):
        reference = replace_special_tokens_and_normalize(label_string[i], vocab_string, processor)
        hypothesis = replace_special_tokens_and_normalize(pred_string[i], vocab_string, processor)
        if reference.replace(" ", "") == "":
            print('Setence:"', label_string[i],'"ignored for the metrics calculate')
            continue
        wer += compute_wer(reference, hypothesis)
        cer += compute_cer(reference, hypothesis)
    if debug:
        print(" > DEBUG: \n\n PRED:", pred_string, "\n Label:", label_string)
    return wer, cer

class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def read_json_with_comments(json_path):
    # fallback to json
    with open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)
    data = json.loads(input_str)
    return data

def load_config(config_path: str) -> AttrDict:
    """Load config files and discard comments

    Args:
        config_path (str): path to config file.
    """
    config = AttrDict()

    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        data = read_json_with_comments(config_path)
    config.update(data)
    return config

def load_vocab(voba_path):
    config = AttrDict()
    config.update(read_json_with_comments(voba_path))
    return config

def save_best_checkpoint(log_dir, model, optimizer, lr_scheduler, scaler, step, epoch, val_loss, best_loss, early_epochs=None):
    if val_loss < best_loss:
        best_loss = val_loss
        if early_epochs is not None:
            early_epochs = 0

        model_save_path = os.path.join(log_dir, 'pytorch_model.bin')
        # model.save_pretrained(log_dir) # export model with transformers for save the config too
        torch.save(model.state_dict(), model_save_path)

        optimizer_save_path = os.path.join(log_dir, 'optimizer.pt')
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'step': step,
            'epoch': epoch
        }

        if scaler is not None:
            checkpoint_dict['scaler'] = scaler.state_dict()

        torch.save(checkpoint_dict, optimizer_save_path)

        print("\n > BEST MODEL ({0:.5f}) saved at {1:}".format(
            val_loss, model_save_path))
    else:
        if early_epochs is not None:
            early_epochs += 1
    return best_loss, early_epochs

transformers.logging.set_verbosity_info()

# wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

def map_data_augmentation(aug_config):
    aug_name = aug_config['name']
    del aug_config['name']
    if aug_name == 'additive':
        return AddBackgroundNoise(**aug_config)
    elif aug_name == 'gaussian':
        return AddGaussianNoise(**aug_config)
    elif aug_name == 'rir':
        return AddImpulseResponse(**aug_config)
    elif aug_name == 'gain':
        return Gain(**aug_config)
    elif aug_name == 'pitch_shift':
        return PitchShift(**aug_config)
    else:
        raise ValueError("The data augmentation '" + aug_name + "' doesn't exist !!")

def evaluation(pred):
    global processor
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    # remove empty strings
    while "" in label_str or " " in label_str:
        if "" in label_str:
            idx = label_str.index("")
            del label_str[idx], pred_str[idx]

        if " " in label_str:
            idx = label_str.index(" ")
            del label_str[idx], pred_str[idx]

    # wer = wer_metric.compute(predictions=pred_str, references=label_str)
    # print("PRED:", pred_str, "Label:", label_str)
    # return {"wer": wer}
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# config_path = 'example/config_example.json'
config = load_config(args.config_path)

OUTPUT_DIR = config['output_path']
os.makedirs(OUTPUT_DIR, exist_ok=True)

vocab = load_vocab(config.vocab['vocab_path'])

if 'preprocess_dataset' in config.keys() and config['preprocess_dataset']:
    def parse_dataset_dict(data_dict):
        text_column = data_dict['text_column']
        audio_path_column = data_dict['path_column']
        del data_dict['text_column']
        del data_dict['path_column']
        return text_column, audio_path_column

    def remove_extra_columns(dataset, text_column, audio_path_column):
        remove_column = list(dataset.column_names)
        remove_column.remove(text_column)
        remove_column.remove(audio_path_column)
        return dataset.remove_columns(remove_column)

    def vocab_to_string(vocab, blank, silence, unk, space=' '):
        vocab_list = list(vocab.keys())
        # remove special tokens
        vocab_list = [x if len(x) == 1 else '' for x in vocab_list]
        vocab_list.sort()
        # remove special with len 1
        vocab_list.remove(silence)
        # vocab_list.remove(blank)
        # vocab_list.remove(unk)

        # append space token
        vocab_list.append(space)
        # convert to string
        return ''.join(vocab_list)


    class Dataset(object):
        def __init__(self, config, vocab, text_column='text', audio_path_column='audio_path'):
            self.config = config
            self.text_column = text_column
            self.audio_path_column = audio_path_column
            self.vocab = vocab
            # load datasets
            self.train_dataset = None
            self.devel_dataset = None

            self.files_path = self.config.datasets['files_path'] if 'files_path'in self.config['datasets'].keys() else None

            self.tokenizer = Wav2Vec2CTCTokenizer(self.config.vocab['vocab_path'], unk_token=self.config.vocab['unk'], pad_token=self.config.vocab['blank'], word_delimiter_token=self.config.vocab['silence'])
            self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=self.config['sampling_rate'], padding_value=0.0, do_normalize=True, return_attention_mask=True)
            self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

            # create dataset
            self.initialize_datasets()


        def preprocess_datasets(self):
            # remove all invalid characters present in text
            self.normalise_texts()

            self.audio_preprocess_and_prepare_dataset()

        def initialize_datasets(self):
            for dataset_dict in self.config.datasets['train']:
                self.train_dataset = self.make_dataset(dataset_dict, self.train_dataset)

            for dataset_dict in self.config.datasets['devel']:
                self.devel_dataset = self.make_dataset(dataset_dict, self.devel_dataset)

        def remove_extra_and_rename_columns(self, dataset, text_column, audio_path_column):
            if 'train' in dataset.keys():
                dataset = dataset['train']
            # remove unused columns
            dataset = remove_extra_columns(dataset, text_column, audio_path_column)

            # rename columns if necessary
            current_column_names = list(dataset.column_names)

            if self.text_column not in current_column_names:
                dataset = dataset.rename_column(text_column, self.text_column)

            if self.audio_path_column not in current_column_names:
                dataset = dataset.rename_column(audio_path_column, self.audio_path_column)
            return dataset

        def make_dataset(self, dataset_dict, own_dataset=None):
            text_column, audio_path_column = parse_dataset_dict(dataset_dict)
            if 'dataset_cache' in self.config and self.config.dataset_cache:
                dataset_dict['cache_dir'] = self.config.dataset_cache

            dataset = load_dataset(**dataset_dict)
            # remove extra columns
            dataset = self.remove_extra_and_rename_columns(dataset, text_column, audio_path_column)

            if own_dataset is None:
                own_dataset = dataset
            else:
                own_dataset = concatenate_datasets([own_dataset, dataset])
            return own_dataset

        def normalise_texts(self):
            vocab_string = vocab_to_string(self.vocab, self.config.vocab['blank'], self.config.vocab['silence'], self.config.vocab['unk'])

            def remove_invalid_characters(batch):
                text = batch[self.text_column].lower()
                text = re.sub("[^{}]".format(vocab_string), " ", text)
                text = re.sub("[ ]+", " ", text)

                batch[self.text_column] = text + " "

                return batch

            print("> Prepare Texts")
            # remove invalid chars
            self.train_dataset = self.train_dataset.map(remove_invalid_characters, num_proc=self.config['num_loader_workers'])
            self.devel_dataset = self.devel_dataset.map(remove_invalid_characters, num_proc=self.config['num_loader_workers'])

        def audio_preprocess_and_prepare_dataset(self):

            def read_audio(batch):
                if self.files_path:
                    batch[self.audio_path_column] = os.path.join(self.files_path, batch[self.audio_path_column])

                speech_array, sampling_rate = torchaudio.load(batch[self.audio_path_column])
                batch["speech"] = speech_array.squeeze().numpy()
                batch["sampling_rate"] = sampling_rate
                batch["target_text"] = batch[self.text_column]
                return batch

            def resample_audio(batch):
                if batch["sampling_rate"] != self.config['sampling_rate']:
                    #speech_array = torchaudio.transforms.Resample(batch["sampling_rate"], self.config['sampling_rate'])(torch.FloatTensor(speech_array).unsqueeze(0)).squeeze().numpy()
                    batch["speech"] = librosa.resample(np.asarray(batch["speech"]),  batch["sampling_rate"], self.config['sampling_rate'])
                    batch["sampling_rate"] = self.config['sampling_rate']
                return batch

            def prepare_dataset(batch):
                try:
                    batch["input_values"] = np.squeeze(self.processor(batch["speech"], sampling_rate=self.config['sampling_rate']).input_values)
                    with self.processor.as_target_processor():
                        batch["labels"] = self.processor(batch["target_text"]).input_ids
                    batch["length"] = len(batch["labels"])
                except:
                    print("Error during load of audio:", batch["target_text"])

                return batch

            print("> Load Audios")
            self.train_dataset = self.train_dataset.map(read_audio, remove_columns=self.train_dataset.column_names)
            self.devel_dataset = self.devel_dataset.map(read_audio, remove_columns=self.devel_dataset.column_names)

            print("> Resample Audios if necessary")
            self.train_dataset = self.train_dataset.map(resample_audio, num_proc=self.config['num_loader_workers'])
            self.devel_dataset = self.devel_dataset.map(resample_audio, num_proc=self.config['num_loader_workers'])

            print("> Prepare dataloader")
            self.train_dataset = self.train_dataset.map(prepare_dataset, remove_columns=self.train_dataset.column_names, num_proc=self.config['num_loader_workers'], batched=False)
            self.devel_dataset = self.devel_dataset.map(prepare_dataset, remove_columns=self.devel_dataset.column_names, num_proc=self.config['num_loader_workers'], batched=False)

    @dataclass
    class DataColletor:
        # Adpated from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """
        def __init__(self, processor, audio_augmentator=None, sampling_rate=16000, padding=True, test=False, max_length=None, max_length_labels=None, pad_to_multiple_of=None, pad_to_multiple_of_labels=None):
            self.processor = processor
            self.audio_augmentator = audio_augmentator
            self.sampling_rate = sampling_rate
            self.padding = padding
            self.test = test
            self.max_length = max_length
            self.max_length_labels = max_length_labels
            self.pad_to_multiple_of = pad_to_multiple_of
            self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = []
            label_features = []
            audio_paths = []
            for feature in features:
                if self.audio_augmentator is not None:
                    input_tensor = self.audio_augmentator(np.array(feature["input_values"]), sample_rate=self.sampling_rate).tolist()
                else:
                    input_tensor = feature["input_values"]

                # input_tensor = feature["input_values"]

                input_features.append({"input_values":input_tensor})
                label_features.append({"input_ids": feature["labels"]})

                if self.test:
                    audio_paths.append(feature['audio_path'])

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels
            if self.test:
                batch["audio_path"] = audio_paths
            return batch
else:
    def parse_dataset_dict(data_dict):
        text_column = data_dict['text_column']
        audio_path_column = data_dict['path_column']
        del data_dict['text_column']
        del data_dict['path_column']
        return text_column, audio_path_column

    def remove_extra_columns(dataset, text_column, audio_path_column):
        remove_column = list(dataset.column_names)
        remove_column.remove(text_column)
        remove_column.remove(audio_path_column)
        return dataset.remove_columns(remove_column)

    def vocab_to_string(vocab, blank, silence, unk, space=' '):
        vocab_list = list(vocab.keys())
        # remove special tokens
        vocab_list = [x if len(x) == 1 else '' for x in vocab_list]
        vocab_list.sort()
        # remove special with len 1
        vocab_list.remove(silence)
        # vocab_list.remove(blank)
        # vocab_list.remove(unk)

        # append space token
        vocab_list.append(space)
        # convert to string
        return ''.join(vocab_list)


    class Dataset(object):
        def __init__(self, config, vocab, text_column='text', audio_path_column='audio_path'):
            self.config = config
            self.text_column = text_column
            self.audio_path_column = audio_path_column
            self.vocab = vocab
            # load datasets
            self.train_dataset = None
            self.devel_dataset = None

            self.files_path = self.config.datasets['files_path'] if 'files_path'in self.config['datasets'].keys() else None

            self.tokenizer = Wav2Vec2CTCTokenizer(self.config.vocab['vocab_path'], unk_token=self.config.vocab['unk'], pad_token=self.config.vocab['blank'], word_delimiter_token=self.config.vocab['silence'])
            self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=self.config['sampling_rate'], padding_value=0.0, do_normalize=True, return_attention_mask=True)
            self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

            # create dataset
            self.initialize_datasets()


        def preprocess_datasets(self):
            # remove all invalid characters present in text
            self.normalise_texts()

            self.audio_preprocess_and_prepare_dataset()

        def initialize_datasets(self):
            for dataset_dict in self.config.datasets['train']:
                self.train_dataset = self.make_dataset(dataset_dict, self.train_dataset)

            for dataset_dict in self.config.datasets['devel']:
                self.devel_dataset = self.make_dataset(dataset_dict, self.devel_dataset)

        def remove_extra_and_rename_columns(self, dataset, text_column, audio_path_column):
            if isinstance(dataset, dict) and 'train' in dataset.keys():
                dataset = dataset['train']
            # remove unused columns
            dataset = remove_extra_columns(dataset, text_column, audio_path_column)

            # rename columns if necessary
            current_column_names = list(dataset.column_names)

            if self.text_column not in current_column_names:
                dataset = dataset.rename_column(text_column, self.text_column)

            if self.audio_path_column not in current_column_names:
                dataset = dataset.rename_column(audio_path_column, self.audio_path_column)
            return dataset

        def make_dataset(self, dataset_dict, own_dataset=None):
            text_column, audio_path_column = parse_dataset_dict(dataset_dict)
            if 'dataset_cache' in self.config and self.config.dataset_cache:
                dataset_dict['cache_dir'] = self.config.dataset_cache

            dataset = load_dataset(**dataset_dict)
            # remove extra columns
            dataset = self.remove_extra_and_rename_columns(dataset, text_column, audio_path_column)

            if own_dataset is None:
                own_dataset = dataset
            else:
                own_dataset = concatenate_datasets([own_dataset, dataset])
            return own_dataset

        def normalise_texts(self):
            vocab_string = vocab_to_string(self.vocab, self.config.vocab['blank'], self.config.vocab['silence'], self.config.vocab['unk'])

            def remove_invalid_characters(batch):
                text = batch[self.text_column].lower()
                text = re.sub("[^{}]".format(vocab_string), " ", text)
                text = re.sub("[ ]+", " ", text)

                batch[self.text_column] = text + " "

                return batch

            print("> Prepare Texts")
            # remove invalid chars
            self.train_dataset = self.train_dataset.map(remove_invalid_characters, num_proc=self.config['num_loader_workers'])
            self.devel_dataset = self.devel_dataset.map(remove_invalid_characters, num_proc=self.config['num_loader_workers'])

        def audio_preprocess_and_prepare_dataset(self):

            def prepare_dataset(batch):
                if self.files_path:
                    batch[self.audio_path_column] = os.path.join(self.files_path, batch[self.audio_path_column])
                batch["input_values"] = batch[self.audio_path_column]
                with self.processor.as_target_processor():
                    batch["labels"] = self.processor(batch[self.text_column]).input_ids
                batch["length"] = len(batch["labels"])
                return batch

            print("> Prepare dataloader")
            self.train_dataset = self.train_dataset.map(prepare_dataset, remove_columns=self.train_dataset.column_names, num_proc=self.config['num_loader_workers'], batched=False)
            self.devel_dataset = self.devel_dataset.map(prepare_dataset, remove_columns=self.devel_dataset.column_names, num_proc=self.config['num_loader_workers'], batched=False)

    @dataclass
    class DataColletor:
        # Adpated from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """
        def __init__(self, processor, audio_augmentator=None, sampling_rate=16000, padding=True, test=False, max_length=None, max_length_labels=None, pad_to_multiple_of=None, pad_to_multiple_of_labels=None):
            self.processor = processor
            self.audio_augmentator = audio_augmentator
            self.sampling_rate = sampling_rate
            self.padding = padding
            self.test = test
            self.max_length = max_length
            self.max_length_labels = max_length_labels
            self.pad_to_multiple_of = pad_to_multiple_of
            self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = []
            label_features = []
            audio_paths = []
            for feature in features:
                try:
                    # load wav
                    speech_array, sampling_rate = torchaudio.load(feature["input_values"])
                    if sampling_rate != self.sampling_rate:
                        raise RuntimeError('Audio Sampling rate different than Config sampling rate ! Make sure that you convert the dataset sampling rate !')
                    speech_array = speech_array.squeeze().numpy()
                    input_tensor = self.processor(speech_array, sampling_rate=sampling_rate).input_values
                    input_tensor = np.squeeze(input_tensor)

                    if self.audio_augmentator is not None:
                        input_tensor = self.audio_augmentator(input_tensor, sample_rate=self.sampling_rate).tolist()

                    input_features.append({"input_values":input_tensor})
                    label_features.append({"input_ids": feature["labels"]})

                    if self.test:
                        audio_paths.append(feature['audio_path'])
                except:
                    print("Error during load of audio:", feature["input_values"])
                    continue

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels
            if self.test:
                batch["audio_path"] = audio_paths
            return batch

    @dataclass
    class DataColletorTest:
        # Adpated from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """
        def __init__(self, processor, sampling_rate=16000, padding=True, test=False, max_length=None, pad_to_multiple_of=None):
            self.processor = processor
            self.sampling_rate = sampling_rate
            self.padding = padding
            self.max_length = max_length
            self.pad_to_multiple_of = pad_to_multiple_of

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = []
            audio_paths = []
            for wav_path in features:
                try:
                # load wav
                    speech_array, sampling_rate = torchaudio.load(wav_path)
                    if sampling_rate != self.sampling_rate:
                        transform = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                        speech_array = transform(speech_array)

                    speech_array = speech_array.squeeze().numpy()
                    input_tensor = self.processor(speech_array, sampling_rate=sampling_rate).input_values
                    input_tensor = np.squeeze(input_tensor)
                    input_features.append({"input_values":input_tensor})
                    audio_paths.append(wav_path)

                except:
                    print("Error during load of audio:", wav_path)
                    continue

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            batch["audio_path"] = audio_paths
            return batch

dataset = Dataset(config, vocab)

# preprocess and normalise datasets
dataset.preprocess_datasets()

processor = dataset.processor

# save the feature_extractor and the tokenizer
processor.save_pretrained(OUTPUT_DIR)

# save vocab
with open(os.path.join(OUTPUT_DIR, 'vocab.json'), "w", encoding="utf-8") as vocab_file:
    json.dump(vocab, vocab_file, ensure_ascii=False)

# save config train
copyfile(args.config_path, os.path.join(OUTPUT_DIR, 'config_train.json'))

# Audio Data augmentation
if 'audio_augmentation' in config.keys():
    from audiomentations import Compose, Gain, AddGaussianNoise, PitchShift, AddBackgroundNoise, AddImpulseResponse
    # ToDo: Implement Time mask and Freq mask
    audio_augmentator = Compose([map_data_augmentation(aug_config) for aug_config in config['audio_augmentation']])
else:
    audio_augmentator = None

# create data colletor
data_collator = DataColletor(processor, audio_augmentator=audio_augmentator, sampling_rate=config.sampling_rate, padding=True)

if os.path.isdir(args.checkpoint_path):
    last_checkpoint = get_last_checkpoint(args.checkpoint_path)
    print("> Resuming Train with checkpoint: ", last_checkpoint)
else:
    last_checkpoint = None

# load model
model = Wav2Vec2ForCTC.from_pretrained(
    last_checkpoint if last_checkpoint else args.checkpoint_path,
    attention_dropout=config['attention_dropout'],
    hidden_dropout=config['hidden_dropout'],
    feat_proj_dropout=config['feat_proj_dropout'],
    mask_time_prob=config['mask_time_prob'],
    layerdrop=config['layerdrop'],
    num_hidden_layers=config['num_hidden_layers'],
    # gradient_checkpointing=config['gradient_checkpointing'], # argumento não é mais aceito. usando model.gradient_checkpointing_enable()
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ctc_zero_infinity=True
)

# freeze feature extractor
if config['freeze_feature_extractor']:
    model.freeze_feature_extractor()

# enable gradient_checkpointing
if config['gradient_checkpointing']:
  model.gradient_checkpointing_enable()
  print("> Grafient checkpointing enabled")

training_args = TrainingArguments(
output_dir=OUTPUT_DIR,
logging_dir=os.path.join(OUTPUT_DIR, "tensorboard"),
report_to="all",
group_by_length=True,
logging_first_step=True,
per_device_train_batch_size=config['batch_size'],
per_device_eval_batch_size=config['batch_size'],
dataloader_num_workers=config['num_loader_workers'],
gradient_accumulation_steps=config['gradient_accumulation_steps'],
seed=config.seed,
num_train_epochs=config['epochs'],
fp16=config.mixed_precision,
logging_steps=config['logging_steps'],
learning_rate=config['lr'],
warmup_steps=config['warmup_steps'],
warmup_ratio=config['warmup_ratio'],
save_strategy="epoch",
evaluation_strategy="epoch",
load_best_model_at_end=config['load_best_model_at_end'],
metric_for_best_model="eval_loss",
greater_is_better=False,
save_total_limit=config['save_total_limit']
)
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=evaluation,
    train_dataset=dataset.train_dataset,
    eval_dataset=dataset.devel_dataset,
    tokenizer=processor.feature_extractor
)

if config['early_stop_epochs']:
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=config['early_stop_epochs']))

print("> Starting Training")
train_result = trainer.train(resume_from_checkpoint=last_checkpoint if args.continue_train else None)
# save best model
# model.save_pretrained(OUTPUT_DIR)
trainer.save_model()

# save train results
metrics = train_result.metrics
metrics["train_samples"] = len(dataset.train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# save eval results
print("--- Evaluate ---")
metrics = trainer.evaluate()
metrics["eval_samples"] = len(dataset.devel_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
