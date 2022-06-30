#usage
#python translate-opus-gen.py input src tgt output 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import sys
from tqdm import *
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

src_lang_id = sys.argv[2]
tgt_lang_id = sys.argv[3]
print(len(sys.argv))
if len(sys.argv) == 5:
        model_name = "Helsinki-NLP/opus-mt-" + str(src_lang_id) + "-" + str(tgt_lang_id)
else:
        model_name = sys.argv[5]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(device)
batch_size=32

translations = []
s = []
c = 0


f = open(sys.argv[1],'r')
num_lines = sum(1 for line in open(sys.argv[1],'r'))
f = open(sys.argv[1],'r')

'''
Following snippet can read file lines on the go. 
'''
for i in tqdm(f.readlines(),total=num_lines):
        if len(i.strip()) > 1:
                s.append(i.strip())
        else:
                s.append('ok')
        c = c + 1 
        if c == batch_size:
                translated = model.generate(**tokenizer(s, return_tensors="pt", padding=True).to(device))
                curr_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                translations = translations + curr_translations
                s = []
                c = 0

translated = model.generate(**tokenizer(s, return_tensors="pt", padding=True).to(device))
curr_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
translations = translations + curr_translations


g = open(sys.argv[4],'w')
for i in translations:
        g.write(i + '\n')

g.close()