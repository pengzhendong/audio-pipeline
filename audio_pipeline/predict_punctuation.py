from funasr import AutoModel

model = AutoModel(model="ct-punc", model_revision="v2.0.4")

fout = open("trans_symbols.txt", "w")
for line in open("/usr/pengzhendong/data/original/pscc/trans.txt"):
    wav, text = line.strip().split("\t", 1)
    if len(text) == 0:
        continue
    else:
        text = model.generate(input=text)[0]["text"]
    fout.write(f"{wav}\t{text}\n")
