stage=0

# can download kaldi obtained fa labels from
# https://univpm-my.sharepoint.com/:u:/g/personal/s1095350_pm_univpm_it/EXQzDQ8PYxVJjOGA2S6hELkBCU33L5WX32eIJ_xupUqXSQ?e=LyXrBk

AMI_ROOT=/media/sam/bx500/amicorpus/audio/ # root to AMI
FALIGN_ROOT=/media/sam/bx500/amicorpus/falign # root to forced alignment data, downloaded or custom
FALIGN_TYPE=tri3 # desired GMM-HMM model used for FA
CONF=conf/train.yml # configuration file (used as labels will be produced for each frame)
LABELS_DIR=/media/sam/bx500/amicorpus/fa_labels # output dir for prepared label files

if [[ $stage -le 0 ]]; then
  echo "Getting diarization from annotation and converting to json files"
  for split in train dev eval
  do
  # if you have fa from KALDI do:
  python local/get_diarization_from_alignments.py $FALIGN_ROOT/$split/${FALIGN_TYPE}a_${split}_ali data/diarization_${split}/${split}.json
  # else use AMI XML data:
  #python local/get_diarization_from_xml.py $FALIGN_ROOT/$split/${FALIGN_TYPE}a_${split}_ali data/diarization_${split}/${split}.json
  python local/get_stats_from_diarization.py data/diarization_${split}/${split}.json >> data/diarization_${split}/stats.txt
  done
fi


# after this you should have a data directory with subfolder with stats and json
# in each json there is speech activity for each speaker in samples !


if [[ $stage -le 1 ]]; then
  echo "Prepping labels"
  for split in train dev eval
  do
  python local/prep_labels.py $AMI_ROOT data/diarization_${split}/${split}.json $CONF $LABELS_DIR/$split
  done
fi

# labels are written in labels_dir as a wave file (so soundfile can be used with start and stop loading
# which is super fast ). however they are vectors containing frame-wise labeling
# for each speaker 1 speech 0 silence so [1, 0, 0, 0] means that in current frame
# spk0 is speaking




