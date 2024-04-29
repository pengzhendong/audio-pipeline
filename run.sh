python extract_wav.py $in_dir $out_dir
python denoise.py $in_dir $out_dir
python loudnorm.py $in_dir $out_dir
python slice.py $in_dir $out_dir
python transcribe_zh.py $in_dir $out_dir
find $out_dir -name "*.txt" -exec cat {} \; > trans.txt
pyton predict_symbol.py trans.txt trans_symbols.txt
