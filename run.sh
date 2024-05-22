in_dir=
out_dir=
separate=
denoise=
wer_threshold=5

stage=0
stop_stage=1

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  if [ "$separate" = true ]; then
    separate="--separate"
  fi
  if [ "$denoise" = true ]; then
    denoise="--denoise"
  fi
  python audio_pipeline/pipeline.py $in_dir $out_dir $separate $denoise
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Perform tanscribing on the audio output by vad instead of the output by noise reduction.
  find $(realpath $out_dir/slices) -name "*.wav" > $out_dir/wav.list
  python audio_pipeline/transcribe.py $out_dir/wav.list --panns --pyannote
  find $out_dir -name "*.paraformer.txt" -exec cat {} \; > $out_dir/paraformer.txt
  find $out_dir -name "*.panns.txt" -exec cat {} \; > $out_dir/panns.txt
  find $out_dir -name "*.pyannote.txt" -exec cat {} \; > $out_dir/pyannote.txt
fi

# use whisper to double check the accuracy of paraformer
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python audio_pipeline/transcribe.py $out_dir/wav.scp --model whisper
  find $out_dir -name "*.whisper.txt" -exec cat {} \; > $out_dir/whisper.txt

  python tools/compute-wer.py \
    --char=1 \
    --v=1 \
    $out_dir/paraformer.txt \
    $out_dir/whisper.txt > $out_dir/trans.wer

  awk -v threshold=$wer_threshold '\
    /utt:/ {utt=$2} \
    /WER:/ {wer=$2} \
    /lab:/ {text=$0; sub(/^lab: /, "", text); gsub(" ", "", text)} \
    # /rec:/ {text=$0; sub(/^rec: /, "", text); gsub(" ", "", text)} \
    /rec:/ && wer>=0 && wer<threshold {print utt "\t" text}' \
    $out_dir/trans.wer > $out_dir/trans.txt
fi
