python -m sockeye.train --source data/toy3/train.en.tok \
    --target data/toy3/train.de.tok \
    --source-graphs data/toy3/train.en.deps \
    --validation-source data/toy3/val.en.tok \
    --validation-target data/toy3/val.de.tok \
    --val-source-graphs data/toy3/val.en.deps \
    --use-grn \
    --grn-type gated \
    --grn-activation tanh \
    --grn-edge-gating \
    --grn-num-layers 2 \
    --grn-num-networks 1 \
    --output toy_model \
    --batch-size 2 \
    --rnn-num-hidden 32 \
    --num-embed 12:32 \
    --grn-num-hidden 32 \
    --checkpoint-frequency 50 \
    --edge-vocab data/toy3/edge_vocab.json \
    --overwrite-output \
    --rnn-attention-num-hidden 18 \
    --grn-dropout 0.5 \
    --grn-norm \
    --grn-positional \
    --grn-pos-embed 12 \
    --skip-rnn
#    --skip-rnn
#    --weight-tying \
#    --weight-tying-type src_trg_softmax
