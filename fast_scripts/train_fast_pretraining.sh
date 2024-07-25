LANG=de

MUSTC_ROOT=/path/data/en-${LANG}
SAVE_DIR=/path/save_models/${LANG}/file_name
pretrain_wav2vec=/path/pretrain_models/wav2vec_small.pt


fairseq-train ${MUSTC_ROOT} \
    --config-yaml config_wave.yaml \
    --train-subset train_wave_joint \
    --valid-subset dev_wave_joint \
    --save-dir ${SAVE_DIR} \
    --max-tokens 3200000  \
    --update-freq 1 \
    --max-update 3200000 \
    --task speech_to_text_wav2vec \
    --criterion label_smoothed_cross_entropy \
    --report-accuracy \
    --arch fast_pretraining \
    --w2v2-model-path ${pretrain_wav2vec}/wav2vec_small.pt \
    --optimizer adam \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 25000 \
    --clip-norm 10.0 \
    --seed 1 \
    --ddp-backend no_c10d \
    --keep-best-checkpoints 10 \
    --patience 15 \
    --max-source-positions 3200000 \
    --skip-invalid-size-inputs-valid-test --fp16 \
    --dropout 0.0 --activation-dropout 0.1 --attention-dropout 0.1 \
    --encoder-layers 8 \
    --encoder-embed-dim 768 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --empty-cache-freq 100 \
    --ignore-prefix-size 1 \
    --tensorboard-logdir ${SAVE_DIR}/tensorboard_logs \
    --best-checkpoint-metric accuracy \
    --maximize-best-checkpoint-metric \


