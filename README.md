# Brazilian Portuguese Quick APT: Quick Automatic Phonetic Transcription for Brazilian Portuguese

Here you can find the data and scripts used to develop my undergraduate thesis titled ["Creating a Dataset for Automatic Phonetic Transcription in Brazilian Portuguese"](https://repositorio.ufjf.br/jspui/handle/ufjf/19250). We used the CORAA ASR corpus [(Candido Junior, 2022)](https://doi.org/10.1007/s10579-022-09621-4) and FalaBrasil's G2P converter [(Neto, 2011)](https://doi.org/10.1007/s13173-010-0023-1) to create a dataset of automatic phonetic transcriptions for training Automatic Phonetic Transcription (APT) models for Brazilian Portuguese (PT-BR). The phonetic transcriptions were standardized according to the phoneme charts presented by ([Ivo, 2019a](https://grad.letras.ufmg.br/arquivos/monitoria/Aula%2002%20apoio.pdf); [Ivo, 2019b](https://grad.letras.ufmg.br/arquivos/monitoria/Aula%2003%20apoio.pdf)), ensuring conformity with the PT-BR phonology literature.

We share the phonetic transcriptions for CORAA ASR's train, dev, and test set, alongside three subsets of the train set (1h, 10h, and 60h of audio), one subset of the dev test (1h of audio), and one subset of the test set (1h of audio). 

### Datasets

The datasets are available at _data\_and\_configs/wav2vec2\_phoneme\_*\_test/input/_. 

### Source data

The CORAA ASR corpus is availabe at [nilc-nlp/CORAA](https://github.com/nilc-nlp/CORAA).

### wav2vec 2.0 models
Furthermore, we fine-tuned three wav2vec 2.0 models, which achieved the following PER (Phonetic Error Rates):

| Subset | Dev    | Test   |
|--------|--------|--------|
| 1h     | 0.8301 | 0.7963 |
| 10h    | 0.2197 | 0.1587 |
| 60h    | 0.2190 | 0.1600 |

Folder _configs/_ contains the configuration files used for fine-tuning. A small description of the input files fields is given below:

| file_path |         g2p        |        g2p_ipa        |              transcript_ipa        |             *transcript_encoded*              |
|-----------|--------------------|-----------------------|--------------------------------|--------------------------------------------|
| File path | Raw G2P transcript | G2P transcript in IPA | Standardized G2P transcript in IPA | *Encoded transcript used in the fine-tuning* |

Additionally, we share the APT model fine-tuned on 10 hours of audio in the [Hugging Face](https://huggingface.co/caiocrocha/wav2vec2-large-xlsr-53-phoneme-portuguese) repository.
