from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import sys
from tqdm import *
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
#tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
#model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
lang_id = sys.argv[2]

model = model.to(device)
batch_size=16
translations = []
s = []
c = 0


f = open(sys.argv[1],'r')
num_lines = sum(1 for line in open(sys.argv[1],'r'))
f = open(sys.argv[1],'r')
for i in tqdm(f.readlines(),total=num_lines):
        if len(i.strip()) > 1:
                s.append(i.strip())
        else:
                s.append('ok')
        c = c + 1 
        if c == batch_size:
                translated = model.generate(**tokenizer(s, return_tensors="pt", padding=True).to(device), forced_bos_token_id=tokenizer.get_lang_id(lang_id))
                curr_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                translations = translations + curr_translations
                s = []
                c = 0

translated = model.generate(**tokenizer(s, return_tensors="pt", padding=True).to(device), forced_bos_token_id=tokenizer.get_lang_id(lang_id))
curr_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
translations = translations + curr_translations


g = open(sys.argv[3],'w')
for i in translations:
        g.write(i + '\n')

g.close()

