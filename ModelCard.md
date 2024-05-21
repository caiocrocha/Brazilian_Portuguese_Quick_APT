# wav2vec2-large-xlsr-53-phoneme-portuguese Model Card

## Model Details

* Developed for a Computational Engineering undergraduate thesis at the Federal University of Juiz de Fora (UFJF), defended by Caio Rocha and supervised by Prof. Jairo Souza.
* Base model: [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53).
* Fine-tuned for Automatic Phonetic Transcription (APT) in Brazilian Portuguese (PT-BR).

## Intended Use

* Intended to be used for APT tasks and for the development of ASR systems.
* Mainly aimed at developers and ASR researchers.
* Not suitable for accent recognition or any task that requires precise phoneme recognition, because the model might confuse phonemes.

## Factors

Potential factors that might affect the model's performance are age, accent, speech rate, fluency, speech idiosyncrasies, background noise, reverberation, and transmission channel phenomena.

## Metrics

* Phone Error Rate (PER) to measure the percentage of phones that have been transcribed incorrectly.
* Accuracy to measure the percentage of correctly predicted phonemes.
* Boxplots of confidence scores by predicted phoneme, aiming to measure the model's level of confidence per class.
* Confusion matrix, providing a detailed view into the most commonly confused phonemes.

## Training Data

* CORAA ASR train set 10 hours sample.
* The audios were randomly selected from the sample that satisfied the following criteria:
    * Portuguese variety: Brazilian Portuguese
    * Number of up votes greater than zero and number of down votes equals zero.
    * Length of transcription is greater than 2.
    * Audio duration is greater than the first quartile (Q1) and lower than the third quartile (Q3), considering the distribution of audio durations.

## Evaluation Data

* CORAA ASR test set 1 hour sample.
* The audios were randomly selected from the sample that satisfied the previously specified criteria.

## Quantitative Analyses

The model was evaluated on the sample test and dev sets, obtaining the following results:

| Set  | PER    | Accuracy |
|------|--------|----------|
| Dev  | 0.2197 | 0.89     |
| Test | 0.1587 | 0.16     |

Additionaly, we evaluated its performance across the phone classes, presented below:

![Boxplot](./images/model_10h/boxplot_phonemes_model_10h.png?raw=true)

![Confusion_matrix](./images/model_10h/confusion_matrix_model_10h.png?raw=true)

## Ethical Considerations

The model is biased towards the accent featured in FalaBrasil's G2P tool, which was used to transcribe the CORAA ASR datasets. Furthermore, even though the corpus contains several PT-BR accents (Recife, Minas Gerais, standard and non-standard São Paulo accents, among others), the model may underperform on speech featuring underrepresented accents.

## Caveats and Recommendations

* To run the model with a dataset, you may use [this script](https://github.com/caiocrocha/Brazilian_Portuguese_Quick_APT/blob/main/scripts/models/simpleTranscription.py).
* The model can be run with either [vocab.json](https://huggingface.co/caiocrocha/wav2vec2-large-xlsr-53-phoneme-portuguese/blob/main/vocab.json) or [encoded_vocab.json](https://huggingface.co/caiocrocha/wav2vec2-large-xlsr-53-phoneme-portuguese/blob/main/encoded_vocab.json), which was originally used for the fine tuning. It contains the same set of phonemes as vocab.json, but in an encoding that has single character keys. For this reason, it can facilitate applications that require forced alignment. To use this encoding, you have to rename encoded_vocab.json to vocab.json.