import math

import torch
from torch import nn
import config


class PositionEncoding(nn.Module):
    def __init__(self, max_len, dim_model):
        super().__init__()
        pe = torch.zeros([max_len, dim_model], dtype=torch.float)
        for pos in range(max_len):
            for _2i in range(0, dim_model, 2):
                pe[pos, _2i] = math.sin(pos / (10000 ** (_2i / dim_model)))
                pe[pos, _2i + 1] = math.cos(pos / (10000 ** (_2i / dim_model)))

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape: [batch_size, seq_len, dim_model]
        seq_len = x.shape[1]
        part_pe = self.pe[0:seq_len]
        # part_pe.shape: [seq_len, dim_model]
        return x + part_pe


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        self.zh_embedding = nn.Embedding(num_embeddings=zh_vocab_size,
                                         embedding_dim=config.DIM_MODEL,
                                         padding_idx=zh_padding_index)

        self.en_embedding = nn.Embedding(num_embeddings=en_vocab_size,
                                         embedding_dim=config.DIM_MODEL,
                                         padding_idx=en_padding_index)

        # 位置编码
        self.position_encoding = PositionEncoding(config.MAX_SEQ_LENGTH, config.DIM_MODEL)

        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                          nhead=config.NUM_HEADS,
                                          num_encoder_layers=config.NUM_ENCODER_LAYERS,
                                          num_decoder_layers=config.NUM_DECODER_LAYERS,
                                          batch_first=True)

        self.linear = nn.Linear(in_features=config.DIM_MODEL, out_features=en_vocab_size)

    def forward(self, src, tgt, src_pad_mask, tgt_mask):
        memory = self.encode(src, src_pad_mask)
        return self.decode(tgt, memory, tgt_mask, src_pad_mask)

    def encode(self, src, src_pad_mask):
        # src.shape = [batch_size, src_len]
        # src_pad_mask.shape = [batch_size, src_len]
        embed = self.zh_embedding(src)
        # embed.shape = [batch_size, src_len, dim_model]
        embed = self.position_encoding(embed)

        memory = self.transformer.encoder(src=embed, src_key_padding_mask=src_pad_mask)
        # memory.shape: [batch_size, src_len, d_model]

        return memory

    def decode(self, tgt, memory, tgt_mask, memory_pad_mask):
        # tgt.shape: [batch_size, tgt_len]
        embed = self.en_embedding(tgt)
        embed = self.position_encoding(embed)
        # embed.shape: [batch_size, tgt_len, dim_model]

        output = self.transformer.decoder(tgt=embed, memory=memory,
                                          tgt_mask=tgt_mask, memory_key_padding_mask=memory_pad_mask)
        # output.shape: [batch_size, tgt_len, dim_model]

        outputs = self.linear(output)
        # outputs.shape: [batch_size, tgt_len, en_vocab_size]
        return outputs