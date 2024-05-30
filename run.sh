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
