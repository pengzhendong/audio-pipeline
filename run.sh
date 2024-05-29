in_dir=
out_dir=
denoise=

stage=0
stop_stage=1

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  if [ "$denoise" = true ]; then
    denoise="--denoise"
  fi
  python audio_pipeline/pipeline.py $in_dir $out_dir $denoise
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Perform tanscribing on the audio output by vad instead of the output by noise reduction.
  python audio_pipeline/transcribe.py $out_dir/slices $out_dir --detect-lang --panns --pyannote
  find $out_dir/asr -name "*.txt" -exec cat {} \; > $out_dir/asr.txt
  find $out_dir/lang -name "*.txt" -exec cat {} \; > $out_dir/lang.txt
  find $out_dir/tags -name "*.txt" -exec cat {} \; > $out_dir/tags.txt
  find $out_dir/speakers -name "*.txt" -exec cat {} \; > $out_dir/speakers.txt
  rm -rf $out_dir/asr $out_dir/lang $out_dir/tags $out_dir/speakers
fi
