import pandas as pd
from tqdm import *
data = pd.read_csv('xnli.15way.orig.tsv',sep='\t')
lang = data.keys()
source = []
reference = []
lp = []
id_ = []
for i in tqdm(lang):
	for j in lang:
		if i==j:
			continue
		curr_lp = i + '-' + j
		source = source + list(data[i])
		reference = reference + list(data[j])
		lp = lp + [curr_lp for k in range(len(data[i]))]
		id_ = id_ + [curr_lp+'-'+str(k) for k in range(len(data[i]))]
dataset = ['xnli' for i in range(len(source))]
df = {'id_':id_,'source': source, 'reference': reference, 'lp':lp,'dataset':dataset}
df = pd.DataFrame(df)
df.to_csv('xnli.15way.source-reference.tsv',sep='\t')