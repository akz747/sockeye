python -m sockeye.train --source data/toy3/train.en.tok \
    --target data/toy3/train.de.tok \
    --validation-source data/toy3/val.en.tok \
    --validation-target data/toy3/val.de.tok \
    --output toy_model \
    --batch-size 2 \
    --rnn-num-hidden 32 \
    --num-embed 12:32 \
    --checkpoint-frequency 50 \
    --overwrite-output \
    --rnn-attention-num-hidden 18 \
#    --skip-rnn
#    --weight-tying \
#    --weight-tying-type src_trg_softmax
