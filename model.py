import torch
import torch.nn as nn
import time


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_num):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.head_dim = embed_dim // head_num
        assert (self.head_dim * head_num == embed_dim), 'embedding size is undeliverable'
        # first we use linear layer for make some kind of copy from input embedding to query, key, and value
        self.value_fc = nn.Linear(embed_dim, embed_dim)
        self.query_fc = nn.Linear(embed_dim, embed_dim)
        self.key_fc = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, key, value, query, mask):
        # because the query key and value shape can be varied due to the part of the code
        # that they are passed the attention, shape should be found using current query key value
        # find batch size number
        batch_size = query.shape[0]
        # find step length
        query_dim = query.shape[1]
        key_dim = key.shape[1]
        value_dim = value.shape[1]

        # apply linearity
        value = self.value_fc(value)  # shape: batch_size, step_length, embedding_dim
        key = self.key_fc(key)  # shape: batch_size, step_length, embedding_dim
        query = self.query_fc(query)  # shape: batch_size, step_length, embedding_dim

        # split embeddings to heads
        query = query.reshape(batch_size, query_dim, self.head_num, self.head_dim)
        key = key.reshape(batch_size, key_dim, self.head_num, self.head_dim)
        value = value.reshape(batch_size, value_dim, self.head_num, self.head_dim)

        # according to attention formula first compute query * key transpose
        # the einsum output shape should be (batch_size,head_num, query_dim, key_dim)
        query_key_multiplication = torch.einsum('nqhd,nkhd->nhqk', [query, key])

        # if there is mask apply it
        # if the mask value was zero should be replaced with very tiny number
        # in order to numerical stability

        if mask is not None:
            query_key_multiplication = query_key_multiplication.masked_fill(
                mask == 0, float("-1e20"))

        # apply softmax to attention matrix
        attention = torch.softmax(query_key_multiplication / (self.embed_dim ** (1 / 2)), dim=3)

        # according to the self attention formula multiply attention with value
        output = torch.einsum('nhql,nlhd->nqhd', [attention, value])

        # einsum output is on shape (batch_size, query_dim, head_num, head_dim)
        # it should be reshaped in order to concat the head_dim and head_num
        output = output.reshape(batch_size, query_dim, self.head_dim * self.head_num)

        # apply last fully connected
        output = self.fc(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, head_num, dropout, expansion_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            head_num=head_num
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, expansion_dim * embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim * expansion_dim, embed_dim)
        )

    def forward(self, value, query, key, mask):
        attention = self.attention(key, value, query, mask)

        x = self.dropout(self.norm1(attention + query))

        feed_forward = self.fc(x)

        return self.dropout(self.norm2(x + feed_forward))


class Encoder(nn.Module):
    def __init__(self,
                 feature_num,
                 embed_dim,
                 seq_len,
                 head_num,
                 dropout,
                 expansion_dim,
                 layer_num,
                 device):
        super(Encoder, self).__init__()
        '''
        :param feature_num: input feature  
        :param embed_dim: dimension that input feature should be mapped
        :param seq_len: input sequence
        :param head_num: number of self-attention head 
        :param dropout: dropout rate
        :param expansion_dim: feed forward mapping dimension 
        :param layer_num: number of encoder layer  
        :param device: cuda or cpu 
        '''
        self.embed_dim = embed_dim
        self.device = device
        self.feature_embedding = nn.Linear(feature_num, embed_dim)
        self.positional_embedding = nn.Embedding(seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    head_num=head_num,
                    expansion_dim=expansion_dim,
                    dropout=dropout
                )
                for _ in range(layer_num)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, mask):
        batch_size, seq_len = seq.shape[0], seq.shape[1]
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(self.device)

        out = self.dropout((self.feature_embedding(seq) + self.positional_embedding(positions)))
        for layer in self.layers:
            # query key and value is the same in encoder
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 head_num,
                 expansion_dim,
                 dropout):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, head_num)
        self.transformer_block = TransformerBlock(
            embed_dim=embed_dim,
            head_num=head_num,
            expansion_dim=expansion_dim,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, in_mask, out_mask):
        attention = self.attention(x, x, x, out_mask)
        query = self.dropout(self.norm(attention + x))
        return self.transformer_block(value, query, key, in_mask)


class Decoder(nn.Module):
    def __init__(self,
                 target_feature,
                 embed_dim,
                 num_layer,
                 head_num,
                 expansion_dim,
                 dropout,
                 device,
                 seq_len):
        super(Decoder, self).__init__()
        self.device = device
        self.feature_embedding = nn.Linear(target_feature, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim,
                    head_num=head_num,
                    expansion_dim=expansion_dim,
                    dropout=dropout
                )
                for _ in range(num_layer)
            ]
        )
        self.fc = nn.Linear(embed_dim, target_feature)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, encoder_output, input_mask, output_mask):
        batch_size, seq_len = seq.shape[0], seq.shape[1]
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(self.device)
        input_seq = self.dropout((self.feature_embedding(seq) + self.position_embedding(positions)))

        for layer in self.layers:
            input_seq = layer(input_seq, encoder_output, encoder_output, input_mask, output_mask)

        return self.fc(input_seq)


class Transformer(nn.Module):
    def __init__(self,
                 input_feature=5,
                 output_feature=2,
                 embed_dim=32,
                 layer_num=4,
                 expansion_dim=4,
                 head_num=4,
                 dropout=0.5,
                 device='cpu',
                 input_seq_len=10,
                 output_seq_len=10):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            feature_num=input_feature,
            embed_dim=embed_dim,
            seq_len=input_seq_len,
            head_num=head_num,
            dropout=dropout,
            expansion_dim=expansion_dim,
            layer_num=layer_num,
            device=device
        )
        self.decoder = Decoder(
            target_feature=output_feature,
            embed_dim=embed_dim,
            num_layer=layer_num,
            head_num=head_num,
            expansion_dim=expansion_dim,
            dropout=dropout,
            device=device,
            seq_len=output_seq_len
        )
        self.device = device

    def make_output_mask(self, seq):
        batch_size, seq_len = seq.shape[0], seq.shape[1]
        mask = torch.tril(torch.ones((seq_len, seq_len))).expand(
            batch_size, 1, seq_len, seq_len
        )

        return mask.to(self.device)

    def forward(self, input_seq, output_seq):
        output_mask = self.make_output_mask(output_seq)
        encoder_output = self.encoder(input_seq, None)
        return self.decoder(output_seq, encoder_output, None, output_mask)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.rand(1000, 10, 5).to(
        device
    )
    trg = torch.rand(1000, 10, 2).to(device)
    model = Transformer(device=device).to(
        device
    )
    start = time.time()
    out = model(x, trg)
    end = time.time()
    print(out.shape)
    print(end - start)
