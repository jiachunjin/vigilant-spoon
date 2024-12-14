from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    bin_dim_l1: int = 32 # 4x4

#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        # xq = apply_rotary_emb(xq, freqs_cis)
        # xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()

    def forward(
        self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(x=self.attention_norm(x), input_pos=start_pos, mask=mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer_bin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)

        self.pos_embedding = nn.Parameter(torch.randn(256*8, config.dim))
        self.tok_eb_level_1 = nn.Linear(2 * 1, config.dim) # 4x4, 2 bits
        self.tok_eb_level_2 = nn.Linear(2 * 1, config.dim) # 8x8, 4 bits
        self.tok_eb_level_3 = nn.Linear(4 * 1, config.dim) # 16x16, 8 bits
        self.tok_eb_level_4 = nn.Linear(8 * 1, config.dim) # 32x32, 16 bits
        # self.tok_eb_level_5 = nn.Linear(8 * 4, config.dim) # 64x64, 24 bits
        # self.tok_eb_level_6 = nn.Linear(8 * 4, config.dim) # 128x128, 32 bits
        # self.tok_eb_level_7 = nn.Linear(32 * 4, config.dim) # 256x256, 64 bits
        self.unit_size = 256

        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output_level_1 = nn.Linear(config.dim, 2 * 1, bias=False)
        self.output_level_2 = nn.Linear(config.dim, 2 * 1, bias=False)
        self.output_level_3 = nn.Linear(config.dim, 4 * 1, bias=False)
        self.output_level_4 = nn.Linear(config.dim, 8 * 1, bias=False)
        # self.output_level_5 = nn.Linear(config.dim, 8 * 4, bias=False)
        # self.output_level_6 = nn.Linear(config.dim, 8 * 4, bias=False)
        # self.output_level_7 = nn.Linear(config.dim, 32 * 4, bias=False)

    def forward(
        self, 
        binary_vec,
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos: Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if binary_vec is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            cond_embeddings = repeat(cond_embeddings, 'b 1 d -> b k d', k=self.unit_size)
            self.cls_token_num = self.unit_size
            token_embeddings = torch.cat([self.tok_eb_level_1(binary_vec[0]),
                                          self.tok_eb_level_2(binary_vec[1]),
                                          self.tok_eb_level_3(binary_vec[2]),
                                          self.tok_eb_level_4(binary_vec[3]),
                                        #   self.tok_eb_level_5(binary_vec[4]),
                                        #   self.tok_eb_level_6(binary_vec[5]),
                                        #   self.tok_eb_level_7(binary_vec[6]),
                                          ], dim=1)

            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
        else:
            raise NotImplementedError("Not implemented yet")
        
        h += self.pos_embedding[:h.shape[1]]

        for layer in self.layers:
            h = layer(h, mask)
        
        # output layers
        h = self.norm(h)
        
        logits_1 = self.output_level_1(h[:, :self.unit_size, :]).float()
        logits_2 = self.output_level_2(h[:, self.unit_size:2*self.unit_size, :]).float()
        logits_3 = self.output_level_3(h[:, 2*self.unit_size:3*self.unit_size, :]).float()
        logits_4 = self.output_level_4(h[:, 3*self.unit_size:4*self.unit_size, :]).float()
        logits_5 = 0
        logits_6 = 0
        logits_7 = 0
        # logits_5 = self.output_level_5(h[:, 4*self.unit_size:5*self.unit_size, :]).float()
        # logits_6 = self.output_level_6(h[:, 5*self.unit_size:6*self.unit_size, :]).float()
        # logits_7 = self.output_level_7(h[:, 6*self.unit_size:7*self.unit_size, :]).float()
        # print(logits_1.shape, logits_2.shape, logits_3.shape)
        logits = [logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7]

        # if self.training:
        #     logits = logits[:, 0:].contiguous()
        
        loss = None
        losses = None
        if valid is not None:
            raise NotImplementedError("Not implemented yet")
        elif targets is not None:
            # logits = logits[:, :-16, :]
            # print(logits.shape, targets.shape)
            loss1 = F.binary_cross_entropy_with_logits(logits_1, targets[0])
            loss2 = F.binary_cross_entropy_with_logits(logits_2, targets[1])
            loss3 = F.binary_cross_entropy_with_logits(logits_3, targets[2])
            loss4 = F.binary_cross_entropy_with_logits(logits_4, targets[3])
            loss5 = torch.tensor(0)
            loss6 = torch.tensor(0)
            loss7 = torch.tensor(0)
            # loss5 = F.binary_cross_entropy_with_logits(logits_5, targets[4])
            # loss6 = F.binary_cross_entropy_with_logits(logits_6, targets[5])
            # loss7 = F.binary_cross_entropy_with_logits(logits_7, targets[6])

            losses = [loss1, loss2, loss3, loss4, loss5, loss6, loss7]

            # loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7) / 7
            loss = (loss1 + loss2 + loss3 + loss4) / 4

            # loss = (
            #     F.binary_cross_entropy_with_logits(logits_1, targets[0]) + 
            #     F.binary_cross_entropy_with_logits(logits_2, targets[1]) +
            #     F.binary_cross_entropy_with_logits(logits_3, targets[2]) +
            #     F.binary_cross_entropy_with_logits(logits_4, targets[3]) +
            #     F.binary_cross_entropy_with_logits(logits_5, targets[4]) +
            #     F.binary_cross_entropy_with_logits(logits_6, targets[5]) + 
            #     F.binary_cross_entropy_with_logits(logits_7, targets[6])
            # ) / 7
            # print('loss', loss)

        return logits, loss, losses
        




if __name__ == '__main__':
    gpt = Transformer_bin(ModelArgs(n_layer=12, n_head=12, dim=768, class_dropout_prob=0.0))

    binary_vec = torch.randint(0, 2, (2, 16, 2), dtype=torch.float32)
    cond_idx = torch.randint(0, 1000, (2,)).long()

    seq_len = 32
    block_size = 16
    num_blocks = seq_len // block_size
    # 构造因果 block mask
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks))  # 下三角矩阵表示因果关系
    block_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    # 转换为布尔类型（True 表示遮挡）
    causal_block_mask = block_mask != 0  # (seq_len, seq_len)
    # 扩展维度以适配 scaled_dot_product_attention
    causal_block_mask = causal_block_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


    targets = torch.rand((2, 16, 2), dtype=torch.float32)
    out = gpt(binary_vec, cond_idx, mask=causal_block_mask, targets=targets)