# Adapting Offline Speech Translation Models for Streaming with Future-Aware Distillation and Inference

Source code for EMNLP 2023 paper: [Adapting Offline Speech Translation Models for Streaming with Future-Aware Distillation and Inference](https://aclanthology.org/2023.emnlp-main.1033.pdf)

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
...
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

We release ours models for streaming inference:

- **en-de**: [checkpoint](http://nlp.xmu.edu.cn/biaofu/models/FAST/FAST_online_en-de.pt) |[config](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-de/config_wave.yaml)|[sentencepiece.model](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-de/spm_unigram10000_st.model)|[sentencepiece.vocab](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-de/spm_unigram10000_st.vocab)|[sentencepiece.txt](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-de/spm_unigram10000_st.txt)
- **en-es**: [checkpoint](http://nlp.xmu.edu.cn/biaofu/models/FAST/FAST_online_en-es.pt) |[config](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-es/config_wave.yaml)|[sentencepiece.model](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-es/spm_unigram10000_st.model)|[sentencepiece.vocab](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-es/spm_unigram10000_st.vocab)|[sentencepiece.txt](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-es/spm_unigram10000_st.txt)
- **en-fr**: [checkpoint](http://nlp.xmu.edu.cn/biaofu/models/FAST/FAST_online_en-fr.pt) |[config](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-fr/config_wave.yaml)|[sentencepiece.model](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-fr/spm_unigram10000_st.model)|[sentencepiece.vocab](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-fr/spm_unigram10000_st.vocab)|[sentencepiece.txt](http://nlp.xmu.edu.cn/biaofu/models/FAST/vocab_en-fr/spm_unigram10000_st.txt)

Streaming Inference

- Vanilla strategy

```shell script
sh fast_scripts/eval_simul_vanilla.sh
```

- FAI strategy

```shell script
sh fast_scripts/eval_simul_fai.sh
```



## Citation

If the paper or the code helps you, please cite the paper in the following format :
```
@inproceedings{fu-etal-2023-adapting,
    title = "Adapting Offline Speech Translation Models for Streaming with Future-Aware Distillation and Inference",
    author = "Fu, Biao and Liao, Minpeng and Fan, Kai and Huang, Zhongqiang and Chen, Boxing and Chen, Yidong and Shi, Xiaodong",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.1033",
    doi = "10.18653/v1/2023.emnlp-main.1033",
    pages = "16600--16619",
}
```