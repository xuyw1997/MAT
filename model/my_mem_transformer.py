import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dCompression(nn.Module):
    """
    ## 1D Convolution Compression $f_c$

    This is a simple wrapper around
    [`nn.Conv1d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
    with some tensor dimension permutations.
    """
    def __init__(self, compression_rate: int, d_model: int, compress_type):
        """
        * `compression_rate` $c$
        * `d_model` is the embedding size
        """
        super().__init__()
        if compress_type == 'conv':
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=compression_rate, stride=compression_rate)
        elif compress_type == 'maxpool':
            self.conv = nn.MaxPool1d(kernel_size=compression_rate, stride=compression_rate)
        else:
            self.conv = nn.AvgPool1d(kernel_size=compression_rate, stride=compression_rate)
        
        
    def forward(self, mem: torch.Tensor):
        """
        `mem` has shape `[seq_len, batch, d_model]`
        """

        # Permute the dimensions of `mem` so that we can run it through the convolution layer.
        # The convolution layer accepts in the form `[batch, features, sequence]`
        mem = mem.permute(0, 2, 1)
        # Get compressed memory by running it through the convolution layer
        c_mem = self.conv(mem)
        # Permute back to form `[seq_len, batch, d_model]`
        return c_mem.permute( 0, 2, 1)

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm



    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)


        return output

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

    def forward_qk(self, q, k, attn_mask=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        qlen, klen, bsz = q.size(1), k.size(1), q.size(0)
        if self.pre_lnorm:
            ##### layer normalization
            q = self.layer_norm(q)

        head_q = self.q_net(q)
        head_k, head_v = torch.chunk(self.kv_net(k), 2, -1)

        head_q = head_q.view(bsz, qlen, self.n_head, self.d_head).transpose(1, 2) 
        head_k = head_k.view(bsz, klen, self.n_head, self.d_head).transpose(1, 2) 
        head_v = head_v.view(bsz, klen, self.n_head, self.d_head).transpose(1, 2) 

        # [qlen x klen x bsz x n_head]
        attn_score = (head_q * self.scale ) @ head_k.transpose(2, 3)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[:,None, None,:] == 0, -float('inf'))
            elif attn_mask.dim() == 3:
                raise

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = attn_prob @ head_v
        attn_vec = attn_vec.transpose(1, 2).contiguous().view(bsz, qlen, -1)
        
        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = q + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(q + attn_out)

        return output

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        """
        :param x: (B, n_head, qlen, k_len)
        """

        bsz, n_head, qlen, k_len = x.shape
        zero_pad = x.new_zeros(bsz, n_head, qlen, 1)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(bsz, n_head, k_len+1, qlen)
        x = x_padded[:, :, 1:, :].view_as(x)

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        self.num_buckets = kwargs.pop('num_buckets', 32)
        self.bidirectional = kwargs.pop('bidirectional', False)
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_head)
        self.relative_attention_bias.weight.data.normal_(mean=0.0, std=1 * ((self.d_model) ** -0.5))





    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        """
            :param w : (B, qlen, d)
            :param r : (klen, d) 
            :param mems: (B, mlen, d)
            :param r_w_bias: (n_head, d_head)
            :param r_r_bias: (n_head, d_head)
            :param attn_mask: 
            klen = mlen + qlen
            d = n_head * d_head
        """
        qlen, rlen, bsz = w.size(1), r.size(0), w.size(0)

        if mems is not None:
            cat = torch.cat([mems, w], 1)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[:, -qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(1)

        w_head_q = w_head_q.view(bsz, qlen, self.n_head, self.d_head).transpose(1, 2)           # bsz x n_head x qlen x d_head
        w_head_k = w_head_k.view(bsz, klen, self.n_head, self.d_head).transpose(1, 2)           # bsz x n_head x klen x d_head
        w_head_v = w_head_v.view(bsz, klen, self.n_head, self.d_head).transpose(1, 2)           # bsz x n_head x klen x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head).transpose(0, 1)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[:, None, :]                             
        AC = rw_head_q @ w_head_k.transpose(2, 3)                               # bsz x n_head x qlen x k_len

        rr_head_q = w_head_q + r_r_bias[:, None, :]
        BD = rr_head_q @ r_head_k.transpose(1, 2).contiguous()

        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, None, ...], -float('inf')).type_as(attn_score)
            else:
                raise

        # bsz x n_head x qlen x k_len
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        # bsz x n_head x qlen x d_head
        attn_vec = attn_prob @ w_head_v

        
        attn_vec = attn_vec.transpose(1, 2).contiguous()
        attn_vec = attn_vec.view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)
        return output
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        memory_position = torch.arange(query_length - key_length, query_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def t5_forward(self, w,  attn_mask=None, mems=None):
        """
            :param w : (B, qlen, d)
            :param mems: (B, mlen, d)
            :param attn_mask: 
            klen = mlen + qlen
            d = n_head * d_head
        """
        qlen, bsz = w.size(1),  w.size(0)

        if mems is not None:
            cat = torch.cat([mems, w], 1)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[:, -qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(1)

        w_head_q = w_head_q.view(bsz, qlen, self.n_head, self.d_head).transpose(1, 2)           # bsz x n_head x qlen x d_head
        w_head_k = w_head_k.view(bsz, klen, self.n_head, self.d_head).transpose(1, 2)           # bsz x n_head x klen x d_head
        w_head_v = w_head_v.view(bsz, klen, self.n_head, self.d_head).transpose(1, 2)           # bsz x n_head x klen x d_head

        #### compute attention score
        attn_score = w_head_q @ w_head_k.transpose(3, 2)
        # [qlen x klen x bsz x n_head]
        attn_score.mul_(self.scale)
        position_bias = self.compute_bias(qlen, klen)
        attn_score += position_bias
        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, None, ...], -float('inf')).type_as(attn_score)
            else:
                raise

        # bsz x n_head x qlen x k_len
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        # bsz x n_head x qlen x d_head
        attn_vec = attn_prob @ w_head_v

        
        attn_vec = attn_vec.transpose(1, 2).contiguous()
        attn_vec = attn_vec.view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)
        return output


    def forward_single_stream(self, clip, text, text_mask, attn_mask=None, mem=None, c_mem=None):
        clip_len, bsz = clip.size(1),  clip.size(0)
        mem_len = mem.size(1) if mem is not None  else 0 
        text_len = text.size(1)

        if mem is not None:
            if c_mem is not None:
                cat = torch.cat([c_mem, mem, clip, text], 1)
                mem_len += c_mem.size(1)
            else:
                cat = torch.cat([mem, clip, text], 1)
        else:
            cat = torch.cat([clip, text], 1)
       
        if self.pre_lnorm:
            w_heads = self.qkv_net(self.layer_norm(cat))
        else:
            w_heads = self.qkv_net(cat)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        
        clip_text_q = w_head_q[:, mem_len:]
        
        
        klen = w_head_k.size(1)
        qlen = clip_len + text_len
        clip_text_q = clip_text_q.view(bsz, qlen, self.n_head, self.d_head).transpose(1, 2)           # bsz x n_head x qlen x d_head
        w_head_k = w_head_k.view(bsz, klen, self.n_head, self.d_head).transpose(1, 2)           # bsz x n_head x klen x d_head
        w_head_v = w_head_v.view(bsz, klen, self.n_head, self.d_head).transpose(1, 2)           # bsz x n_head x klen x d_head


         #### compute attention score
        attn_score = clip_text_q @ w_head_k.transpose(3, 2)
        # [qlen x klen x bsz x n_head]
        attn_score.mul_(self.scale)
        position_bias = self.compute_bias(clip_len, mem_len + clip_len)
        attn_score[:, :, :clip_len, :mem_len + clip_len] += position_bias

        #### compute attention probability

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, ...].expand(bsz, -1, -1)
                text_mask = (text_mask == 0)
                clip_text_mask = text_mask[:, None, :].expand(-1, clip_len, -1)
                mm_mask = torch.cat([attn_mask, clip_text_mask], dim=-1)
                text_text_mask = text_mask[:, None, :].expand(-1, text_len, -1)
                text_text_mask = torch.cat([text_mask.new_zeros(bsz, text_len, mem_len + clip_len).bool(), text_text_mask], dim=-1)
                mm_mask = torch.cat([mm_mask, text_text_mask], dim=1)
                
                attn_score = attn_score.float().masked_fill(
                    mm_mask[:, None, ...], -float('inf')).type_as(attn_score)
            else:
                raise

        # bsz x n_head x qlen x k_len
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        # bsz x n_head x qlen x d_head
        attn_vec = attn_prob @ w_head_v

        
        attn_vec = attn_vec.transpose(1, 2).contiguous()
        attn_vec = attn_vec.view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = cat[:, mem_len:] + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(cat[:, mem_len:] + attn_out)

        return output
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_head, self.d_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_self_bias(self, query_length, key_length, relative_attention_bias, bidirectional=False, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        memory_position = torch.arange(query_length - key_length, query_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=bidirectional,
            num_buckets=self.num_buckets
        )
        values = relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def compute_cross_bias(self, query_length, key_length, relative_attention_bias):
        device = relative_attention_bias.weight.device
        position_id = torch.arange(key_length, dtype=torch.long, device=device)[None, :].expand(query_length, -1)
        values = relative_attention_bias(position_id)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values


        
    
class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

    def forward_qk(self, q, k, attn_mask=None):
        output = self.dec_attn.forward_qk(q, k, attn_mask)
        output = self.pos_ff(output)
        return output

class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, compress_type, compression_rate, 
                 num_buckets=32, bidirectional=False,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                                         num_buckets=num_buckets, bidirectional=bidirectional, **kwargs)

        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))
        if compression_rate > 1:
            self.compress = Conv1dCompression(compression_rate, d_model, compress_type)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn.t5_forward(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output
    
    
    def forward_single_stream(self, clip, txt, txt_mask, dec_attn_mask=None, mem=None, c_mem=None):
        
        output = self.dec_attn.forward_single_stream(clip, txt, txt_mask, dec_attn_mask, mem, c_mem)
        
        output = self.pos_ff(output)

        return output
    


class MemTransformerLM(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, pre_lnorm=False,
                 tgt_len=None, ext_len=0, mem_len=None, 
                 same_length=False, attn_type=0, clamp_len=-1,
                 compress_type='conv', compress_rate=4, c_mem_len=0, num_buckets=32,
                 bidirectional=False
                 ):
        super(MemTransformerLM, self).__init__()
        
        d_embed = d_model 
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len
        self.c_mem_len = c_mem_len
        self.compression_rate = compress_rate
        self.attn_type = attn_type
        self.bidirectional = bidirectional

        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        compress_type=compress_type, compression_rate=self.compression_rate,
                        num_buckets=num_buckets, bidirectional=bidirectional)
                )
        elif attn_type == 1: # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        self.same_length = same_length
        self.clamp_len = clamp_len





    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len


    def _update_mems(self, hids, mems, c_mems):

        if mems is not None:
            assert len(hids) == len(mems), 'len(hids) != len(mems)'

        if self.mem_len == 0:
            assert mems[0] is None 
            assert c_mems[0] is None 
            return mems, c_mems
        
        new_mems = []
        for i in range(len(hids)):
            cat = torch.cat([mems[i], hids[i]], dim=1) if mems[i] is not None  else hids[i]
            new_mems.append(cat.detach())
        new_c_mems = c_mems

        new_mem_len = new_mems[0].shape[1]
        if new_mem_len > self.mem_len:
            # n_c_mem = (new_mem_len - self.mem_len + self.compression_rate - 1) // self.compression_rate
            # n_old = n_c_mem * self.compression_rate

            # A list to keep memories that need to be compressed for each layer.
            mem_to_compress = []
            # A list to keep the memories that do not get compressed for each layer.
            uncompressed_mem = []
            # Iterate through memories of each layer.
            for m in new_mems:
                # Split the memories at $c n_{cm}$
                # cm, m = torch.split(m, [n_old, new_mem_len - n_old], dim=1)
                cm, m = torch.split(m, [new_mem_len - self.mem_len, self.mem_len], dim=1)
                # Collect memories to compress
                mem_to_compress.append(cm)
                # Collect remaining memories
                uncompressed_mem.append(m)
            # Update the memories
            new_mems = uncompressed_mem

            cur_c_mems = []
            if self.c_mem_len > 0:
                for i, layer in enumerate(self.layers):
                    cur_c_mems.append(layer.compress(mem_to_compress[i]))

                if c_mems[0] is not None:
                    new_c_mems = [torch.cat((m.detach(), nm), dim=1) for m, nm in zip(c_mems, cur_c_mems)]
                # If there are no old compressed memories
                else:
                    new_c_mems = cur_c_mems
                # Truncate old memories
                if new_c_mems[0].shape[1] > self.c_mem_len:
                    new_c_mems = [m[:, -self.c_mem_len:] for m in new_c_mems]
                
            
        return new_mems, new_c_mems
    
    def init_mems(self):
        
        mems = []
        for i in range(self.n_layer+1):
            mems.append(None)

        c_mems = []
        for i in range(self.n_layer+1):
            c_mems.append(None)

        return mems, c_mems

    def _forward(self, clip, mems=None):
        """
        :param: clip: B, T, d
        """
        bsz, qlen = clip.size()[:2]

        mlen = mems[0].size(1) if mems[0] is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = clip.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool() # -1
        else:
            dec_attn_mask = torch.triu(
                clip.new_ones(qlen, klen), diagonal=1+mlen).bool()

        hids = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=clip.device, 
                                   dtype=clip.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(clip)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)

            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems
    

    def forward(self, clip, *, mems):
        """
        :param: clip: B, T, d
        """
        if not mems: mems = self.init_mems()
        hidden, new_mems = self._forward(clip, mems=mems)

        
        return hidden,  new_mems
    

    def _forward_clip_text(self, clip, txt, txt_mask, mems=None):
        """
        :param: clip: B, T, d
        """
        bsz, qlen = clip.size()[:2]

        mlen = mems[0].size(1) if mems[0] is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = clip.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool() # -1
        else:
            dec_attn_mask = torch.triu(
                clip.new_ones(qlen, klen), diagonal=1+mlen).bool()

        hids = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=clip.device, 
                                   dtype=clip.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(clip)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer.forward_clip_txt(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, txt, txt_mask, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward_clip_text(self, clip, txt, txt_mask, *, mems):
        """
        :param: clip: B, T, d
        """
        if not mems: mems = self.init_mems()
        hidden, new_mems = self._forward_clip_text(clip, txt, txt_mask, mems=mems)

        return hidden,  new_mems

    def _forward_single_stream(self, clip, txt, txt_mask, mems=None, c_mems=None):
        """
        :param: clip: B, T, d
        """
        bsz, qlen = clip.size()[:2]

        mlen = mems[0].size(1) if mems[0] is not None else 0
        c_mlen = c_mems[0].size(1) if c_mems[0] is not None else 0
        mlen += c_mlen
        klen = mlen + qlen
        if self.same_length:
            all_ones = clip.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool() # -1
        else:
            if self.bidirectional:
                dec_attn_mask = clip.new_zeros(qlen, klen).bool()
            else:
                dec_attn_mask = torch.triu(clip.new_ones(qlen, klen), diagonal=1+mlen).bool()

        hids = []
        if self.attn_type == 0: # default
            
            core_out = self.drop(clip)
            hids.append(core_out)
            clip_len = core_out.size(1)
            txt_len = txt.size(1)
            for i, layer in enumerate(self.layers):
                # ablation  ernie
                output = layer.forward_single_stream(core_out, txt, txt_mask, dec_attn_mask=dec_attn_mask, mem=mems[i], c_mem=c_mems[i])

                assert output.size(1) == clip_len + txt_len
                core_out = output[:, :clip_len]
                txt = output[:, clip_len:]
                hids.append(core_out)
        

        core_out = self.drop(core_out)

        new_mems, new_c_mems = self._update_mems(hids, mems, c_mems)
        
        return core_out, txt, new_mems, new_c_mems
    
    def forward_single_stream(self, clip, txt, txt_mask, *, mems, c_mems):
        """
        :param: clip: B, T, d
        """
        if not mems: 
            mems, c_mems = self.init_mems()
        hidden, txt, new_mems, new_c_mems = self._forward_single_stream(clip, txt, txt_mask, mems=mems, c_mems=c_mems)

        
        return hidden,  txt, new_mems, new_c_mems