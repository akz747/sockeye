python -m sockeye.train --source data/toy2/train.en.tok \
    --target data/toy2/train.de.tok \
    --source-graphs data/toy2/train.en.deps \
    --validation-source data/toy2/val.en.tok \
    --validation-target data/toy2/val.de.tok \
    --val-source-graphs data/toy2/val.en.deps \
    --use-cpu \
    --use-grn \
    --grn-edge-gating \
    --grn-num-layers 4 \
    --output toy_model \
    --batch-size 2 \
    --rnn-num-hidden 30 \
    --num-embed 32 \
    --grn-num-hidden 17 \
    --checkpoint-frequency 50 \
    --edge-vocab data/toy2/edge_vocab.json \
    --overwrite-output \
    --rnn-attention-num-hidden 18 \
#    --skip-rnn
#    --weight-tying \
#    --weight-tying-type src_trg_softmax
