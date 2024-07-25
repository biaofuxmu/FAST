beam=$1
LANG=$2
cif=$3

MUSTC_ROOT=/path/data/en-${LANG}
SAVE_DIR=/path/save_models/${LANG}/filename

if [ -z "$cif" ]; then
  task=speech_to_text_wav2vec
  dev_data_name=dev_wave_joint
else
  task=speech_to_text_wav2vec_cif
  dev_data_name=dev_wavecif_joint
fi


CHECKPOINT_FILENAME=avg_best_10_ckpt.pt
infer_results=${SAVE_DIR}/offline_results/avg_best_10_ckpt_beam${beam}
model=${SAVE_DIR}/${CHECKPOINT_FILENAME}
if [ ! -f ${model} ];then
  ckpt=$((ckpt+5))
  python scripts/average_checkpoints.py \
    --inputs ${SAVE_DIR} \
    --num-epoch-checkpoints 10 \
    --checkpoint-upper-bound ${ckpt} \
    --output ${model}
fi


subset="tst-COMMON_wavecif_joint"

python airseq_cli/generate.py ${MUSTC_ROOT} \
    --config-yaml config_wave.yaml \
    --gen-subset ${subset} \
    --task ${task} \
    --path ${model} \
    --max-tokens 3200000 \
    --beam ${beam} \
    --scoring sacrebleu \
    --max-source-positions 3200000 \
    --prefix-size 1 \
    --results-path ${infer_results} \
    #--fp16

tail -1 ${infer_results}/generate-${subset}.txt