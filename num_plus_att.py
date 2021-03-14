import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import argparse
import pandas as pd
pd.set_option('display.width',None)
from pathlib import Path


parser = argparse.ArgumentParser(
    description='Create literals'
)
parser.add_argument('--dataset', default='YAGO3-10', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`} to be used? (default: YAGO3-10)')
args = parser.parse_args()


# Load vocab
vocab = np.load(f'{str(Path.home())}/.data/FB15k-237/vocab_e1', allow_pickle=True)

ent2idx = vocab[0]
idx2ent = vocab[1]

# Load raw literals
df = pd.read_csv(f'data/FB15k-237/literals/numerical_literals.txt', header=None, sep='\t')

rel2idx = {v: k for k, v in enumerate(df[1].unique())}

# Resulting file
num_lit = np.zeros([len(ent2idx), len(rel2idx)], dtype=np.float32)

# Create literal wrt vocab

str_sum = ''
with open('data/FB15k-237/literals/att_txt.txt') as f:
    for line in f.readlines():
        line_lit = line.split('\t')
        str_sum += str(line_lit[1])
str_sum = str_sum.replace('\n','')
str_list = str_sum.split(' ')
if str_list:
    str_list_single = list(set(str_list))
att2idx = {}
for v in range(len(str_list_single)):
    att2idx[str_list_single[v]] = v

att_vec = np.zeros([len(ent2idx), len(att2idx)], dtype=np.float32)
with open('data/FB15k-237/literals/att_txt.txt') as f1:
    for line1 in f1.readlines():
        line_lit1 = line1.split('\t ')
        line_lit1[1] = line_lit1[1].replace('\n', '')
        line_lit1[1] = line_lit1[1].split(' ')
        for att in line_lit1[1]:
            att_vec[ent2idx[line_lit1[0].lower()],att2idx[att]] = 1

num_lit_att = [[] for i in range(len(ent2idx))]
print(num_lit_att)
for i, (s, p, lit) in tqdm(enumerate(df.values)):
    if s.lower() in ent2idx:
        num_lit[ent2idx[s.lower()], rel2idx[p]] = lit
        num_lit_att[ent2idx[s.lower()]] =  [*num_lit[ent2idx[s.lower()]], *att_vec[ent2idx[s.lower()]]]

for i in range(len(num_lit_att)):
    if len(num_lit_att[i])==0:
         num_lit_att[i] = np.zeros(len(rel2idx)+len(att2idx), dtype=np.float32)
for i in range(10):
    print('****************************************************')
    print(num_lit_att[i])
    print(len(num_lit_att[i]))

print(type(num_lit_att[0][0]))
np.save(f'data/FB15k-237/literals/num_plus_att.npy', num_lit_att)