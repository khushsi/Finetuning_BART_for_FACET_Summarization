TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=1024
UPDATE_FREQ=4
BART_PATH=model.pt

sleep 1m

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin \
    --save-dir checkpoints \
    --restore-file checkpoints/model.pt \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --batch-size 1 \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 2500 \
    --max-update	$TOTAL_NUM_UPDATES \
    --find-unused-parameters;


#!/bin/bash
echo "different checkpoints"
sleep 30
for entry in `ls checkpoints/*.pt | sed -r 's/^.+\///'`
do
echo $entry
`sh emerald_evaluate.sh $entry`
done
