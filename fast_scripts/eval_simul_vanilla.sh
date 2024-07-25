lagging=9
lang=de

MUSTC_ROOT=/path/data/en-${lang}
SRC_LIST_OF_AUDIO=${MUSTC_ROOT}/data/tst-COMMON/test_data/tst-COMMON.wav_list
TGT_FILE=${MUSTC_ROOT}/data/tst-COMMON/test_data/tst-COMMON.${lang}
ST_SAVE_DIR=/path/save_models/${lang}/file_name/online_ckpt_avg10
CHECKPOINT_FILENAME=checkpoint_online.pt
WAV2VEC_DIR=/path/pretrain_models/wav2vec_small.pt
OUTPUT=${ST_SAVE_DIR}/online_results/vanilla/wait${lagging}/

port=12346

simuleval --agent ./examples/speech_to_text/simultaneous_translation/agents/fast_agent.py \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${MUSTC_ROOT} \
    --config config_wave.yaml \
    --model-path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --w2v2-model-path ${WAV2VEC_DIR} \
    --output ${OUTPUT} \
    --waitk-lagging ${lagging} \
    --fixed-pre-decision-ratio 1 \
    --max-source-positions 3200000 \
    --fast-cif \
    --scores \
    --port ${port} \
    --gpu \
    --lang ${lang} \
    --simul-mode vanilla \
