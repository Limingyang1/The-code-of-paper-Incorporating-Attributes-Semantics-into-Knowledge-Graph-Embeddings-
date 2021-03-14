For DistMult/ComplEx + LiteralE-AN
python main_literal.py dataset {FB15k, FB15k-237} model {DistMult, ComplEx} input_drop 0.2 embedding_dim 100 batch_size 128 epochs 200 lr 0.001 process True  ### Should make sure line115 in main_literal.py uses numerical_literals.npy

For DistMult/ComplEx + LiteralE-AT
python main_literal.py dataset {FB15k, FB15k-237} model {DistMult_text, ComplEx_text} input_drop 0.2 embedding_dim 100 batch_size 128 epochs 200 lr 0.001 process True

For DistMult/ComplEx + LiteralE-Combine
python main_literal.py dataset {FB15k, FB15k-237} model {DistMult_text, ComplEx_text} input_drop 0.2 embedding_dim 100 batch_size 128 epochs 200 lr 0.001 process True  ### Should make sure line115 in main_literal.py uses num_plus_att.npy

For ConvE + LiteralE-AN
python main_literal.py dataset {FB15k, FB15k-237} model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 embedding_dim 200 batch_size 128 epochs 150 lr 0.001 process True ### Should make sure line115 in main_literal.py uses numerical_literals.npy

For ConvE + LiteralE-AT
python main_literal.py dataset {FB15k, FB15k-237} model ConvE_text input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 embedding_dim 200 batch_size 128 epochs 150 lr 0.001 process True

For ConvE + LiteralE-Combine
python main_literal.py dataset {FB15k, FB15k-237} model ConvE_text input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 embedding_dim 200 batch_size 128 epochs 150 lr 0.001 process True ### Should make sure line115 in main_literal.py uses num_plus_att.npy

For DistMult_glin/ComplEx_glin + LiteralE-AN
python main_literal.py dataset {FB15k, FB15k-237} model {DistMult_glin, ComplEx_glin} input_drop 0.2 embedding_dim 100 batch_size 128 epochs 200 lr 0.001 process True  ### Should make sure line115 in main_literal.py uses numerical_literals.npy

For DistMult_glin/ComplEx_glin + LiteralE-AT
python main_literal.py dataset {FB15k, FB15k-237} model {DistMult_glin_text, ComplEx_glin_text} input_drop 0.2 embedding_dim 100 batch_size 128 epochs 200 lr 0.001 process True

For DistMult_glin/ComplEx_glin + LiteralE-Combine
python main_literal.py dataset {FB15k, FB15k-237} model {DistMult_glin_text, ComplEx_glin_text} input_drop 0.2 embedding_dim 100 batch_size 128 epochs 200 lr 0.001 process True  ### Should make sure line115 in main_literal.py uses num_plus_att.npy

For ConvE_glin + LiteralE-AN
python main_literal.py dataset {FB15k, FB15k-237} model ConvE_glin input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True ### Should make sure line115 in main_literal.py uses numerical_literals.npy

For ConvE_glin + LiteralE-AT
python main_literal.py dataset {FB15k, FB15k-237} model ConvE_glin_text input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True

For ConvE_glin + LiteralE-Combine
python main_literal.py dataset {FB15k, FB15k-237} model ConvE_glin_text input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True ### Should make sure line115 in main_literal.py uses num_plus_att.npy