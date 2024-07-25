LANG=de
MUSTC_ROOT=/path/data/en-${LANG}
SAVE_DIR=/path/save_models/${LANG}/file_name
pretrain_wav2vec=/path/pretrain_models/wav2vec_small.pt
pretrain_model_2nd=/path/save_models/${LANG}/file_name/avg_best_10_ckpt.pt  


fairseq-train ${MUSTC_ROOT} \
    --config-yaml config_wave.yaml \
    --train-subset train_wavecif_joint \
    --valid-subset dev_wavecif_joint \
    --save-dir ${SAVE_DIR} \
    --max-tokens 3200000 \
    --update-freq 1 \
    --max-update 3200000 \
    --task speech_to_text_wav2vec_cif \
    --criterion fkd_ce_acc \
    --arch fast \
    --w2v2-model-path ${pretrain_wav2vec} \
    --optimizer adam \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --clip-norm 10.0 \
    --seed 1 \
    --ddp-backend no_c10d \
    --keep-best-checkpoints 10 \
    --patience 15 \
    --max-source-positions 3200000 \
    --skip-invalid-size-inputs-valid-test \
    --dropout 0.0 --activation-dropout 0.1 --attention-dropout 0.1 \
    --encoder-layers 8 \
    --encoder-embed-dim 768 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --ignore-prefix-size 1 --log-interval 100 \
    --load-pretrained-encoder-from ${pretrain_model_2nd} \
    --load-pretrained-decoder-from ${pretrain_model_2nd} \
    --tensorboard-logdir ${SAVE_DIR}/tensorboard_logs \
    --num-workers 0 \
    --lambda-w2v2 1.0 \
    --lambda-cif 1.0 \
    --fp16 \
    --best-checkpoint-metric loss \
    --future-mask-length 50

