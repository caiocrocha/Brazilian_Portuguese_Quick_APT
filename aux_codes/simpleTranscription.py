#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from numpy import array
import soundfile as sf
from argparse import ArgumentParser
import os
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()


# Adapted from https://github.com/huggingface/transformers/tree/main
def chunk_iter(
    processor: Wav2Vec2Processor,
    device: torch.device,
    speech: array,
    sample_rate: int,
    chunk_len: int,
    stride_left: int,
    stride_right: int,
) -> dict:
    """Iterate over chunk of audio

    Args:
        speech (array): audio data
        sample_rate (int): audio sample rate
        chunk_len (int): length of the chunk in frames
        stride_left (int): left stride of the chunk in frames
        stride_right (int): right stride of the chunk in frames

    Yields:
        Iterator[dict]: result of the processing
    """
    speech_len = speech.shape[0]
    step = chunk_len - stride_left - stride_right
    for i in range(0, speech_len, step):
        # add start and end paddings to the chunk
        chunk = speech[i : i + chunk_len]
        processed = processor(
            chunk, sampling_rate=sample_rate, return_tensors="pt"
        ).to(device)
        _stride_left = 0 if i == 0 else stride_left
        is_last = i + step + stride_left >= speech_len
        _stride_right = 0 if is_last else stride_right
        if chunk.shape[0] > _stride_left:
            yield {
                "is_last": is_last,
                "stride": (chunk.shape[0], _stride_left, _stride_right),
                **processed,
            }

def get_logits(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: torch.device,
    speech: array,
    sample_rate: int,
    chunk_length_s: float = 0,
    stride_length_s: float = None,
) -> array:
    """Compute the logits of an audio using the model

    Args:
        speech (array): audio data
        sample_rate (int): audio sample rate
        chunk_length_s (float, optional): compute the logits by splitting the audio in chunks,
            which might reduce compute time. If value is zero, then the audio isn't split. Defaults to 0.
        stride_length_s (float, optional): overlap between the chunks. Defaults to None.

    Raises:
        ValueError: if the chunk length is shorter than the stride length
        ValueError: if the chunk length or the stride length are negative

    Returns:
        array: logits
    """
    speech_len = speech.shape[0]

    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6

    if isinstance(stride_length_s, (int, float)):
        stride_length_s = [stride_length_s, stride_length_s]

    if chunk_length_s == 0:
        chunk_length_s = speech_len

    align_to = model.config.inputs_to_logits_ratio
    chunk_len = int(round(chunk_length_s * sample_rate / align_to)) * align_to
    stride_left = int(round(stride_length_s[0] * sample_rate / align_to)) * align_to
    stride_right = (
        int(round(stride_length_s[1] * sample_rate / align_to)) * align_to
    )

    if chunk_len < stride_left + stride_right:
        raise ValueError("Chunk length must be longer than stride length")

    if chunk_len < 0 or stride_left < 0 or stride_right < 0:
        raise ValueError("Chunk length and stride length must be non-negative")

    model.to(device)
    logits_list = []
    for model_inputs in chunk_iter(
        processor, device, speech, sample_rate, chunk_len, stride_left, stride_right
    ):
        input_n, left, right = model_inputs.pop("stride")
        input_values = model_inputs.pop("input_values")
        attention_mask = model_inputs.pop("attention_mask", None)
        with torch.no_grad():
            outputs = model(
                input_values=input_values, attention_mask=attention_mask
            )

        logits = outputs.logits
        if left > 0 or right > 0:
            ratio = 1 / model.config.inputs_to_logits_ratio
            token_n = int(round(input_n * ratio))
            left = int(round(left / input_n * token_n))
            right = int(round(right / input_n * token_n))
            right_n = token_n - right
            logits = logits[:, left:right_n, :]
        logits_list.append(logits)

    model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    return torch.concat(logits_list, axis=1)

def transcribe(file_path, model, processor, device):
    if os.path.isfile(file_path):
        sf_audio = sf.SoundFile(file_path)
        speech, sample_rate= (
            sf_audio.read(),
            sf_audio.samplerate,
        )

        logits = get_logits(
                    model,
                    processor,
                    device,
                    speech,
                    sample_rate,
                    chunk_length_s=0,
                )

        transcript = processor.batch_decode(
                        torch.argmax(logits, dim=-1)
                    )[0]
        return transcript
    else:
        print(f'No such file: {file_path}!')
        return None

def get_cmd_line():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help="Input CSV file with the audio paths")
    parser.add_argument('-m', '--model', required=True, type=Path, help="Model directory path")
    parser.add_argument('-o', '--output', required=True, help='Output transcript in CSV')
    return parser.parse_args()

def main():
    args = get_cmd_line()

    df = pd.read_csv(args.input)
    df.head()

    if os.path.isfile(args.model / 'pytorch_model.bin'):
        from_flax = False
    else:
        from_flax = True

    model = Wav2Vec2ForCTC.from_pretrained(args.model, from_flax=from_flax)
    processor = Wav2Vec2Processor.from_pretrained(args.model, from_flax=from_flax)

    vocab = processor.tokenizer.get_vocab()
    print(f"Vocab: {vocab}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df['transcript'] = df['file_path'].progress_apply(transcribe, args=(model,processor,device))
    print(len(df['transcript']))
    df[['file_path','transcript']].to_csv(args.output, index=False)

if __name__=='__main__': main()
