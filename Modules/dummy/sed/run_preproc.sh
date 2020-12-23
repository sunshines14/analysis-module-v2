#!/bin/bash
  
nj=8

basedir=/workspace/Modules/dummy/sed

# Convert to WAV
ffmpeg -y -i $1 -acodec pcm_s16le -ac 1 -ar 16000 $basedir/input/input.wav || exit 1;

# Preprocessing
python3 $basedir/sed_preproc.py $basedir/input/input.wav || exit 1;
find $basedir/files -name '*.wav' | sort > $basedir/input/files.txt || exit 1;
