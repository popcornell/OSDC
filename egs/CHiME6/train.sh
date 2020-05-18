gpus=0
export CUDA_VISIBLE_DEVICES=$gpus
EXP_DIR=exp/crnn_augm
CONFS=conf/train.yml

mkdir -p $EXP_DIR
cp -r local $EXP_DIR/code
python local/train_crnn.py $CONFS $EXP_DIR $gpus