import torch
from torch import nn
import math

from torch.utils import data
from data_load import load_data, load_cn_vocab, load_en_vocab
from data_my_Transformer import en2cn_dataset
import os, sys
import numpy as np
import csv

from torch.utils.tensorboard import SummaryWriter

import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout_p , seq_max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.posi_enc = torch.zeros((1, seq_max_len, hidden_size))
        tmp = torch.arange(0, seq_max_len, dtype=torch.float32).reshape(-1, 1)
        tmp2 = torch.pow(10000, torch.arange(0, hidden_size, 2, dtype=torch.float32)/hidden_size)
        tmp3 = tmp / tmp2
        self.posi_enc[:, :, 0::2] = torch.sin(tmp3)     # 能用pytorch中矩阵操作完成的，就不要自己写for循环
        self.posi_enc[:, :, 1::2] = torch.cos(tmp3)
    def forward(self, X):   # 注意！这里的X的形状和上面说的(n,d)不太一样，这里多了一个batch维度。
        X = X + self.posi_enc[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)  # dropout防止过拟合
    
class PositionwiseFFN(nn.Module):
    def __init__(self, input_h_size, ffn_h_size, ffn_output_size):
        super(PositionwiseFFN, self).__init__()
        self.dense1 = nn.Linear(input_h_size, ffn_h_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_h_size, ffn_output_size)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def musked_softmax(self, scores, valid_lens):
        """
        score: [Batch_size, Num of quaries, Num of keys]
        其中Num of keys这个维度是被查询文本的序列长度l，
        valid_len在这个维度上指定有效长度
        """
        def _seq_mask(scores, valid_lens, value=0):
            max_seq_len = scores.size(1)    # num_of_keys
            mask = torch.arange(max_seq_len,
                                dtype=torch.float32, device=scores.device)[None, :] < valid_lens[:, None]
            scores[~mask] = value
            return scores
        
        if valid_lens is None:
            return nn.functional.softmax(scores, dim=-1)
        else:
            qk_shape = scores.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, qk_shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)

            scores = _seq_mask(scores.reshape(-1, qk_shape[-1]), valid_lens, value=1e-6)
            return nn.functional.softmax(scores.reshape(qk_shape), dim=-1)

    
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[2]
        alpha = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = self.musked_softmax(alpha, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, attention_fun, num_hiddens, num_heads, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = attention_fun
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def reshape_qkv(self, X):
        shape = X.shape
        X = X.reshape(shape[0], shape[1], self.num_heads, -1)

        # 将X转换为(batch_size, num_heads, q的数目或者k-v的数目——文本长度, p_q or p_k or p_v)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])    # return前的这一步相当于把所有的头连接在一起
    
    def reshape_attention_output(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)    # (batch_size, q的数目, p_o)

    def forward(self, quaries, keys, values, valid_lens):
        """
        输入的qkv的形状是[batch_size, q的数目或者k-v的数目——文本长度, d_q或者d_k或d_v]
        映射后的qkv的维度是p_q, p_k, p_v
        """
        q = self.reshape_qkv(self.W_q(quaries))   # (batch_size * num_heads, q的数目, p_q)
        k = self.reshape_qkv(self.W_k(keys))
        v = self.reshape_qkv(self.W_v(values))

        if valid_lens is not None:
            v_l = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            output = self.attention(q, k, v, v_l)  # output.shape = (batch_size * num_heads, q的数目, p_o/h)
            output_concat_heads = self.reshape_attention_output(output)
            return self.W_o(output_concat_heads)

class LayerNorm_man(nn.Module):
    def __init__(self, num_hiddens, eps=1e-12):
        super(LayerNorm_man, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_hiddens))
        self.beta = nn.Parameter(torch.zeros(num_hiddens))
        self.eps = eps
    
    def forward(self, X):
        mean = X.mean(-1, keepdim=True)
        var = X.var(-1, unbiased=False, keepdim=True)   # 在最后一个维度，也就是num_hiddens上计算
        out = (X - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta      # why?

        return out          
class AddNorm(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm_man(embedding_dim)

    def forward(self, X, Y):
        output = self.layer_norm(self.dropout(Y) + X)
        return output
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_heads, num_hiddens, dropout, ffn_h_size, bias=False):
        super(TransformerEncoderBlock, self).__init__()
        attention_fun = DotProductAttention(dropout)
        self.attention = MultiHeadAttention(attention_fun, num_hiddens, num_heads, bias)
        self.positionwise_ffn = PositionwiseFFN(num_hiddens, ffn_h_size, num_hiddens)
        self.add_norm1 = AddNorm(num_hiddens, dropout)
        self.add_norm2 = AddNorm(num_hiddens, dropout)
    
    def forward(self, X, valid_len):
        Y = self.add_norm1(X, self.attention(X, X, X, valid_len))
        output = self.add_norm2(Y, self.positionwise_ffn(Y))
        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, num_hiddens, seq_max_len, dropout, ffn_h_size,
                 num_blks, vocab_size, pad_idx=None, bias=False):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, num_hiddens, padding_idx=pad_idx)
        self.position_ecd = PositionalEncoding(num_hiddens, dropout, seq_max_len)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module('blok'+str(i), 
                                 TransformerEncoderBlock(num_heads, num_hiddens, 
                                                         dropout, ffn_h_size, bias))
            
    def forward(self, X, valid_lens):
        X = self.position_ecd(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = []
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights.append(blk.attention.attention.attention_weights)
        return X

class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_heads, num_hiddens,
                 ffn_h_size, dropout, i, bias=False):
        super(TransformerDecoderBlock, self).__init__()
        self.i = i
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        attention_fun = DotProductAttention(dropout)
        self.attention1 = MultiHeadAttention(attention_fun, num_hiddens, num_heads, bias)
        self.add_norm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(attention_fun, num_hiddens, num_heads, bias)
        self.add_norm2 = AddNorm(num_hiddens, dropout)
        self.positionwise_ffn = PositionwiseFFN(num_hiddens, ffn_h_size, num_hiddens)
        self.add_norm3 = AddNorm(num_hiddens, dropout)
    
    def forward(self, X, enc_output, enc_valid_len):
        """
        X.shape: [batch_size, seq_len_after_padding = max_seq_len, embedding_size]
        """
        if self.training:
            batch_size = X.shape[0]
            seq_len_after_padding = X.shape[1]
            dec_valid_len = torch.arange(1, seq_len_after_padding + 1,
                                         device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_len = None

        # 1. compute self attention
        decoder_layer_input = X
        self_atten = self.attention1(quaries=X, keys=X, values=X, valid_lens=dec_valid_len)

        # 2. first add and norm
        output_add_norm1 = self.add_norm1(decoder_layer_input, self_atten)

        if enc_output is not None:
            # 3. compute encoder-decoder attention
            e_d_atten = self.attention2(quaries=output_add_norm1, keys=enc_output,
                                        values=enc_output, valid_lens=enc_valid_len)
            # 4. second add and norm
            output_add_norm2 = self.add_norm2(output_add_norm1, e_d_atten)
        
        # 5. positionwise feed forward network
        output_pos_ffn = self.positionwise_ffn(output_add_norm2)

        # 6. third add and norm
        output_add_norm3 = self.add_norm3(output_add_norm2, output_pos_ffn)

        return output_add_norm3
    
class TransformerDecoder(nn.Module):
    def __init__(self, num_heads, num_hiddens,
                 seq_max_len,
                 ffn_h_size, dropout, num_blks,
                 vocab_size, pad_idx=None, bias=False):
        super(TransformerDecoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout, seq_max_len)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i),
                                 TransformerDecoderBlock(num_heads=num_heads,
                                                         num_hiddens=num_hiddens,
                                                         ffn_h_size=ffn_h_size,
                                                         dropout=dropout,
                                                         i=i))

    def forward(self, X, enc_output, enc_valid_len):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for _, blk in enumerate(self.blks):
            X = blk(X, enc_output, enc_valid_len)

        return X

class MyTransformer(nn.Module):
    def __init__(self, num_heads, num_hiddens, seq_max_len,
                 ffn_h_size, dropout, num_blks_enc, num_blk_dec,
                 vocab_size_src, vocab_size_tar,
                 pad_idx=None, bias=False):
        super(MyTransformer, self).__init__()
        self.pad_idx = pad_idx
        self.encoder = TransformerEncoder(num_heads, num_hiddens, seq_max_len, dropout, ffn_h_size,
                                          num_blks_enc, vocab_size_src, pad_idx, bias)
        self.decoder = TransformerDecoder(num_heads, num_hiddens, seq_max_len, ffn_h_size, dropout,
                                          num_blk_dec, vocab_size_tar, pad_idx, bias)
        self.output_layer = nn.Linear(num_hiddens, vocab_size_tar)
        
    def forward(self, srcX, tarX, enc_valid_len, contex_time_step):
        enc_output = self.encoder(srcX, enc_valid_len)

        # 这里的decoder没有传入vlaid_lens是因为decoder block里面生成了，这样不好，需要优化！
        dec_output = self.decoder(tarX, enc_output, enc_valid_len)
        output = self.output_layer(dec_output)

        return output

if __name__ == '__main__':
    # Config
    BATCH_SIZE = 64
    LR = 0.001
    NUM_HIDDENS = 512
    FFN_H_SIZE = 2048
    N_LAYERS = 6
    NUM_HEADS = 8
    DROPOUT_RATE = 0.2
    N_EPOCH = 60
    PAD_ID = 0
    TRAIN_SET_PROP = 0.8
    MAX_SEQ_LEN = 50

    device = 'cuda'
    log_path = './log'
    model_save_path = './model_save'
    writer = SummaryWriter(log_path)
    cn2idx, idx2cn = load_cn_vocab()    # len = 12946
    en2idx, idx2en = load_en_vocab()    # len = 11370
    model = MyTransformer(NUM_HEADS, NUM_HIDDENS, MAX_SEQ_LEN, FFN_H_SIZE, DROPOUT_RATE,
                        N_LAYERS, N_LAYERS, len(en2idx), len(cn2idx), PAD_ID).to(device)
    print(">>> Model prepared.")
    
    with not any(os.listdir(model_save_path)):
        X_train, Y_train,\
        Source, Target,\
        cn_valid_lens, en_valid_lens = load_data('train') # X:cn, Y:en | [seq_num, emb_size_after_padding]
        contex_time_step = [None] * N_LAYERS
        dataset = en2cn_dataset(X_train, cn_valid_lens, Y_train, en_valid_lens)
        train_data_loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)
        print(">>> Train Data prepared.")


        model.train()
        optimizer = torch.optim.Adam(model.parameters(), LR)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        print(">>> Criterion prepared.")
        loss = 0.0
        print(">>> Train start.")
        for epoch_i in range(N_EPOCH):
            print("  >>> epoch {}".format(epoch_i))
            batch_i = 0
            for data_batch in train_data_loader:
                batch_i = batch_i + 1
                en_batch, en_valid_lens_batch,\
                    cn_batch, cn_valid_lens_batch = data_batch
                en_batch = en_batch.to(device)
                en_valid_lens_batch = en_valid_lens_batch.to(device)
                cn_batch = cn_batch.to(device)
                cn_valid_lens_batch = cn_valid_lens_batch.to(device)
                batch_size, seq_len_after_padding = cn_batch.shape
                optimizer.zero_grad()
                cn_hat = model(en_batch, cn_batch,
                            en_valid_lens_batch, contex_time_step)   # [64, 50, 12946]
                
                cn_hat_trans = cn_hat.view(batch_size*seq_len_after_padding, -1)  # 
                cn_batch_trans = cn_batch.view(batch_size*seq_len_after_padding, -1).squeeze()
                
                loss = criterion(cn_hat_trans, cn_batch_trans)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            print(f"loss = {loss}")

            writer.add_scalar('loss', loss, epoch_i)
        # Save trained model.
        torch.save(model.state_dict(), model_save_path)

    with any(os.listdir(model_save_path)):
        model.load_state_dict(torch.load(model_save_path))
        X_test, Y_test,\
        Source, Target,\
        cn_valid_lens, en_valid_lens = load_data('test') # X:cn, Y:en | [seq_num, emb_size_after_padding]
        dataset = en2cn_dataset(X_test, cn_valid_lens, Y_test, en_valid_lens)
        test_data_loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)
        # Load vocabularies
        cn_vocab = {}
        en_vocab = {}
        vocab_files = ['./data/cn.txt.vocab.txt', './data/en.txt.vocab.txt']
        for file_path in vocab_files:
            _, file_name = os.path.split(file_path)
            with open(file_path, 'r', newline='') as file:
                tsv_reader = csv.reader(file, delimiter='\t')
                for row in tsv_reader:
                    if len(row) >= 2:
                        key = row[1]
                        value = row[0]
                        if (file_name[0:2] == 'cn'):
                            assert file_name[0:2] == 'cn'
                            cn_vocab[key] = value
                        else:
                            assert file_name[0:2] == 'en'
                            en_vocab[key] = value

        model.eval()
        print (">>> Test start.")
        with torch.no_grad():
            for test_batch in test_data_loader:
                en_batch, en_valid_lens_batch,\
                            cn_batch, cn_valid_lens_batch = test_batch

        

"""
  >>> epoch 55
loss = 2.8752103389706463e-05
  >>> epoch 56
loss = 2.6463276299182326e-05
  >>> epoch 57
loss = 2.7921922082896344e-05
  >>> epoch 58
loss = 2.324249544471968e-05
  >>> epoch 59
loss = 1.834472277550958e-05
"""