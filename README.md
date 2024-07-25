# Adapting Offline Speech Translation Models for Streaming with Future-Aware Distillation and Inference

Source code for EMNLP 2023 paper: Adapting Offline Speech Translation Models for Streaming with Future-Aware Distillation and Inference

## Data Processing
Take German for example.
Firstly, download [MuST-C v1.0](https://ict.fbk.eu/must-c/) archive `MUSTC_v1.0_en-de.tar.gz` to the `${MUSTC_ROOT}` path, and uncompress it:

```shell script
LANG=de
MUSTC_ROOT=/path/data/en-${LANG}$
tar -xzvf MUSTC_v1.0_en-de.tar.gz
```
Then, run the script to prepare data manifest.
```shell script
python3 examples/speech_to_text/prep_mustc_data_raw.py --data-root ${MUSTC_ROOT} \
  --tgt-lang ${LANG}
```

The generated `.tsv` should be expanded with the field of source language text and doubled with asr task. Here's some examples from the `.tsv` file.

```
id      audio   n_frames        tgt_text        speaker tgt_lang        src_text        src_lang
ted_2529_66     /xxx/en-de/data/train/wav/ted_2529.wav:9517120:61760      61760   Ich hatte den Vorteil einer Perspektive von dieser Breite.  spk.2529        de      I had the benefit of a spectrum this wide.      en
ted_1257_134    /xxx/en-de/data/train/wav/ted_1257.wav:13876160:80960     80960   And outside the library, I wanted to make a place to cultivate your mind.   spk.1257        en      And outside the library, I wanted to make a place to cultivate your mind.       en
ted_362_30      /xxx/en-de/data/train/wav/ted_362.wav:488959:156960       156960  Ich lebe genau hier im West Village, die Rauchwolke wurde zum Glück westwärts geweht, weg von uns.  spk.362 de      I live right there in the West Village, so the plume was luckily blowing west, away from us.        en
...
ted_526_7       /xxx/en-de/data/train/wav/ted_526.wav:16538720:19360      19360   It can also happen in the brain.    spk.526 en      It can also happen in the brain.        en
ted_190_62      /xxx/en-de/data/train/wav/ted_190.wav:7045920:47360       47360   Simple question: if you can't read and write, how do you manage your contact information?   spk.190 en      Simple question: if you can't read and write, how do you manage your contact information?   en
ted_1771_81     /xxx/en-de/data/train/wav/ted_1771.wav:9624320:25600      25600   This is my message to you. spk.1771 en      This is my message to you.      en
```

The preprocessed directory `${MUSTC_ROOT}` should look like as follows:

```
.
├── en-de
│   ├── config_wave.yaml
│   ├── data
│   ├── dev_wavecif_joint.tsv
│   ├── docs
│   ├── segment
│   ├── spm_unigram10000_st.model
│   ├── spm_unigram10000_st.txt
│   ├── spm_unigram10000_st.vocab
│   ├── train_wavecif_joint.tsv
│   ├── tst-COMMON_wavecif_joint.tsv
│   ├── tst-HE_wavecif_joint.tsv
└── MUSTC_v1.0_en-de.tar.gz
```

The generated `config_wave.yaml` should look like as follows:

```
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: spm_unigram10000_st.model
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
vocab_filename: spm_unigram10000_st.txt
use_audio_input: true
prepend_tgt_lang_tag: true
```

## Training

+ Train an offline model with multitask pretraining.

```shell script
sh fast_scripts/train_fast_pretraining.sh
```

+ Train an offline model with CIF module.

```shell script
sh fast_scripts/train_fast_offline.sh
```

+ Train an offline model with FKD.
```shell script
sh fast_scripts/train_fast.sh
```

## Evaluation
### Offline Translation

```shell script
sh fast_scripts/eval_offline.sh
```

### Streaming Translation

Note that the offline models need to be converted to support streaming translation task. 
```shell script
sh fast_scripts/convert_fast_offline_to_online.sh
```
or
```shell script
sh fast_scripts/convert_fast_to_online.sh
```

+ Vanilla strategy
```shell script
sh fast_scripts/eval_simul_vanilla.sh
```

+ FAI strategy
```shell script
sh fast_scripts/eval_simul_fai.sh
```
