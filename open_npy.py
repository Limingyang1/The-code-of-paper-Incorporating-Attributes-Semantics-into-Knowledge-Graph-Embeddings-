import numpy as np
np.set_printoptions(threshold=np.inf)

test = np.load(r'data/FB15k/literals/numerical_literals.npy')
print(test[:10])

'''
vocab = np.load(f'{str(Path.home())}/.data/{args.dataset}/vocab_e1', allow_pickle=True)

ent2idx = vocab[0]
idx2ent = vocab[1]

print(ent2idx)
print(inx2ent)
'''