stage=1

AMI_ROOT=/home/sam/Desktop/amicorpus/
FALIGN_ROOT=/home/sam/Desktop/amicorpus/falign
FALIGN_TYPE=tri3
CONF=conf/train.yml
LABELS_DIR=$AMI_ROOT/labels

if [[ $stage -le 0 ]]; then
  echo "Get diarization"
  for split in train dev eval
  do
  python local/get_diarization_from_alignments.py $FALIGN_ROOT/$split/${FALIGN_TYPE}a_${split}_ali data/diarization_${split}/${split}.json
  python local/get_stats_from_diarization.py data/diarization_${split}/${split}.json >> data/diarization_${split}/stats.txt
  done
fi


if [[ $stage -le 1 ]]; then
  for split in train dev eval
  do
  python local/prep_labels.py $AMI_ROOT data/diarization_${split}/${split}.json $CONF $LABELS_DIR/$split
  done
fi



