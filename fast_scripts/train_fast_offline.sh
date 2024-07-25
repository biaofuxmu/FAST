LANG=de
MUSTC_ROOT=/path/data/en-${LANG}
SAVE_DIR=/path/save_models/${LANG}/file_name
pretrain_wav2vec=/path/pretrain_models/wav2vec_small.pt
pretrain_model_1st=/path/save_models/${LANG}/file_name/avg_best_10_ckpt.pt


fairseq-train ${MUSTC_ROOT} \
    --config-yaml config_wave.yaml \
    --train-subset train_wavecif_joint \
    --valid-subset dev_wavecif_joint \
    --save-dir ${SAVE_DIR} \
    --task speech_to_text_wav2vec_cif \
    --arch fast_offline \
    --criterion qua_ce_acc \
    --max-tokens 3200000 --max-update 3200000 \
    --max-source-positions 3200000 \
    --update-freq 1 --num-workers 4 \
    --optimizer adam --lr 0.0001 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --seed 1 --clip-norm 10.0 \
    --keep-best-checkpoints 10 \
    --patience 15 --ddp-backend no_c10d \
    --skip-invalid-size-inputs-valid-test \
    --w2v2-model-path ${pretrain_wav2vec} \
    --load-pretrained-encoder-from ${pretrain_model_1st} \
    --load-pretrained-decoder-from ${pretrain_model_1st} \
    --dropout 0.0 --activation-dropout 0.1 --attention-dropout 0.1 \
    --encoder-layers 8 \
    --encoder-embed-dim 768 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --ignore-prefix-size 1 --log-interval 100 --fp16 \
    --tensorboard-logdir ${SAVE_DIR}/tensorboard_logs \
    --maximize-best-checkpoint-metric \
    --eval-bleu --eval-bleu-args '{"beam": 5, "prefix_size": 1}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe sentencepiece \
    --best-checkpoint-metric bleu \


