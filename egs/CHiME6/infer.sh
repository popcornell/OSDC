gpus=0
export CUDA_VISBLE_DEVICES=$gpus
EXP_DIR=exp/crnn_augm
WAVS=/media/sam/bx500/amicorpus/audio/
CKPT=_ckpt_epoch_21.ckpt
OUT=$EXP_DIR/preds/${CKPT}

python $EXP_DIR/code/infer_sequential.py $EXP_DIR $CKPT $WAVS $OUT $gpus --regex Array1-01