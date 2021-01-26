gpus=0
export CUDA_VISBLE_DEVICES=$gpus
EXP_DIR=exp/tcn
WAVS=/media/sam/bx500/amicorpus/audio/
CKPT=checkpoints-v3.ckpt
OUT=$EXP_DIR/preds/${CKPT}

python $EXP_DIR/code/infer.py $EXP_DIR $CKPT $WAVS $OUT $gpus --regex Array1-01