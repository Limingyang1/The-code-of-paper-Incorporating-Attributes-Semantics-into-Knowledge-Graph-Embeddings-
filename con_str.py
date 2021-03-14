from tqdm import tqdm
import numpy as np
import pandas as pd
pd.set_option('display.width',None)
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Create literals'
)
parser.add_argument('--dataset', default='YAGO3-10', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`} to be used? (default: YAGO3-10)')
args = parser.parse_args()

vocab = np.load(f'{str(Path.home())}/.data/FB15k-237/vocab_e1', allow_pickle=True)
ent2idx = vocab[0]
idx2ent = vocab[1]

df = pd.read_csv(f'data/FB15k-237/literals/numerical_literals.txt', header=None, sep='\t')
str = ['' for i in range(len(ent2idx))]
s_lit = []
with open('data/FB15k-237/literals/att_txt.txt','a') as file_handle:
    for i, (s, p, lit) in tqdm(enumerate(df.values)):
        p = p.replace('http://rdf.freebase.com/ns/', '')
        p = p.replace('http://www.w3.org/2000/01/rdf/', '')
        p = p.replace('topic_server.', '')
        p = p.replace('.', ' ')
        p = p.replace('_', ' ')
        if s.lower() in ent2idx:
            if str[ent2idx[s.lower()]].find(p)<0:
                str[ent2idx[s.lower()]] += '/' + p
                str_ori = str[ent2idx[s.lower()]]
                a = str_ori.replace('http://rdf.freebase.com/ns/', '')
                a = a.replace('http://www.w3.org/2000/01/rdf/', '')
                a = a.replace('topic_server.', '')
                a = a.replace('.', ' ')
                a = a.replace('_', ' ')
                a = a.replace('/', ' ')
                str[ent2idx[s.lower()]] = a
            if s not in s_lit:
                s_lit.append(s)
                #print(s_lit)
                file_handle.write(s)
                file_handle.write('\t')
                file_handle.write(a)
                file_handle.write('\n')

