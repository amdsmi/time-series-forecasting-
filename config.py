import torch

# transformer hyperparameter
input_feature = 6
output_feature = 6
embed_dim = 32
layer_num = 4
expansion_dim = 4
head_num = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_seq_len = 10
output_seq_len = 10


# dataset hyperparameter


# train hyperparameter
learning_rate = 3e-4
epochs = 100