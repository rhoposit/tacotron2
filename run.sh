



# TRAINING command to run
DATASET=vqcodes
DATA=/home/smg/v-j-williams/workspace/external_modified/data/sys5_txt_mixed
CHECKPOINTS=/home/smg/v-j-williams/workspace/external_modified/sys5_checkpoints
HPARAM=hparams_modified.py

train.py --dataset=$DATASET --data-root=$DATA --checkpoint-dir=$CHECKPOINTS --hparams=$HPARAM


# INFERENCE command to run
#synthesize.py  --dataset=$DATASET --data-root=$DATA --checkpoint-dir=$CHECKPOINTS --postnet-checkpoint-dir=</path/to/postnet/model/dir> --hparams=$HPARAM
