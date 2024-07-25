lang=de
MUST_ROOT=/path/data
SAVE_DIR=/path/save_models/${lang}/file_name
ONLINE_PATH=${SAVE_DIR}/online_ckpt_avg10 
CHECKPOINT=avg_best_10_ckpt.pt 

ARCH=fast_simul

python fairseq_cli/convert_online_model.py \
    ${MUST_ROOT}/en-${lang} \
    --user-dir examples/simultaneous_translation \
    --config-yaml config_wave.yaml \
    --save-dir ${ONLINE_PATH} \
    --num-workers 8 \
    --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
    --criterion label_smoothed_cross_entropy  \
    --warmup-updates 4000 --max-update 100000 \
    --max-tokens 3200000 --seed 1 \
    --load-pretrained-encoder-from ${SAVE_DIR}/${CHECKPOINT} \
    --load-pretrained-decoder-from ${SAVE_DIR}/${CHECKPOINT} \
    --w2v2-model-path /path/pretrain_models/wav2vec_small.pt \
    --task speech_to_text_wav2vec_cif \
    --arch ${ARCH} \
    --simul-type waitk_fixed_pre_decision  \
    --waitk-lagging 30 --fixed-pre-decision-ratio 1 \
    --update-freq 8 \
    --report-accuracy --log-format json  \
    --log-interval 10 --save-interval-updates 1 \
    --ignore-prefix-size 1 \
    --simul-mode fai \
    --encoder-layers 8 \
    --encoder-embed-dim 768 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
