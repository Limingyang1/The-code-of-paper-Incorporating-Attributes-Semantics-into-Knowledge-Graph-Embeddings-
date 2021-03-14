import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import argparse
import pandas as pd
pd.set_option('display.width',None)
import spacy
from pathlib import Path
import unicodedata


parser = argparse.ArgumentParser(
    description='Create text literals'
)
parser.add_argument('--dataset', default='YAGO3-10', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`} to be used? (default: YAGO3-10)')
args = parser.parse_args()

vocab = np.load(f'{str(Path.home())}/.data/{args.dataset}/vocab_e1', allow_pickle=True)

ent2idx = vocab[0]
idx2ent = vocab[1]

# Load raw literals
df_num = pd.read_csv(f'data/{args.dataset}/literals/numerical_literals.txt', header=None, sep='\t')
df_num10 = df_num[:10]

rel2idx = {v: k for k, v in enumerate(df_num[1].unique())}

# Resulting file
d_emb = 300
num_lit = np.zeros([len(ent2idx), len(rel2idx)], dtype=np.float32)


# Load preprocessor
nlp = spacy.load('en_core_web_md')

att_lit = np.zeros([len(ent2idx), d_emb * (len(rel2idx))], dtype=np.float32)
cnt = 0

# Create literal wrt vocab

for i, (s, p, lit) in tqdm(enumerate(df_num.values)):
    num_lit[ent2idx[s.lower()], rel2idx[p]] = lit
    a = p.replace('http://rdf.freebase.com/ns/', '')
    a1 = a.replace('.', ' ')
    a2 = a1.replace('_', ' ')
    att_lit[ent2idx[s.lower()],rel2idx[p] * d_emb : (rel2idx[p]+1) * d_emb] = [x*float(lit) for x in nlp(a2).vector]
    #print('******************************************') 
    #print(att_lit[ent2idx[s.lower()],rel2idx[p] * d_emb : (rel2idx[p]+1) * d_emb])
    #print(att_lit[ent2idx[s.lower()]])

#print(att_lit[ent2idx['/m/0n5c9'.lower()],:10*300])

np.save(f'data/FB15k/literals/att_literals.npy', att_lit)
