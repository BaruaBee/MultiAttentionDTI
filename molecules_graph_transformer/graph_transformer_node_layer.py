import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
import math

"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available


def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] + edges.data[explicit_edge])}
    return func


def add_spatial_pos(implicit_attn, spatial_pos):
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] + edges.data[spatial_pos])}
    return func

# To copy edge features to be passed to FFN_e


def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func

# 新增：添加相对位置编码


def get_relative_positions(seq_len):
    x = torch.arange(seq_len).unsqueeze(0)
    y = torch.arange(seq_len).unsqueeze(1)
    return (x - y).float()

# 新增：自注意力中的缩放点积函数


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention




class NodeGraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer，只使用节点特征和空间位置信息，不需要显式的边特征输入
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1, layer_norm=True, batch_norm=True, residual=True, use_bias=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # 节点特征变换
        self.Q = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim, bias=use_bias)

        # 空间位置编码变换
        self.proj_spatial_pos = nn.Linear(in_dim, out_dim, bias=use_bias)

        # 输出投影
        self.O_h = nn.Linear(out_dim, out_dim)

        # 注意力dropout
        self.attention_dropout = nn.Dropout(0.1)

        # 门控机制
        self.gate = nn.Linear(in_dim + out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

        # 改进的FFN
        self.FFN_h = nn.Sequential(
            nn.Linear(out_dim, out_dim*4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(out_dim*4, out_dim)
        )

    def propagate_attention(self, g):
        # 计算注意力分数
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))

        # 缩放
        g.apply_edges(scaling('score', np.sqrt(
            self.out_channels // self.num_heads)))

        # 添加空间位置信息
        g.apply_edges(add_spatial_pos('score', 'proj_spatial_pos'))

        # softmax
        g.apply_edges(exp('score'))

        # 添加注意力dropout
        g.edata['score'] = self.attention_dropout(g.edata['score'])

        # 向目标节点发送加权值
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge(
            'V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge(
            'score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, spatial_pos):
        h_in1 = h  # 第一个残差连接

        # 多头注意力计算
        # 1. 线性变换
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # 2. 处理空间位置信息
        proj_spatial_pos = self.proj_spatial_pos(spatial_pos)

        # 3. 重塑为[num_nodes, num_heads, feat_dim]形式
        batch_size = h.shape[0] // g.number_of_nodes()
        num_nodes = g.number_of_nodes() if batch_size == 1 else g.number_of_nodes() // batch_size

        head_dim = self.out_channels // self.num_heads

        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, head_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, head_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, head_dim)
        g.edata['proj_spatial_pos'] = proj_spatial_pos.view(
            -1, self.num_heads, head_dim)

        # 4. 传播注意力
        self.propagate_attention(g)

        # 5. 处理注意力输出
        h_attn_out = g.ndata['wV'] / \
            (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))

        # 6. 使用门控机制
        concat = torch.cat([h.view_as(h_attn_out), h_attn_out], dim=-1)
        gate_val = torch.sigmoid(
            self.gate(concat.view(h.size(0), -1)).view_as(h_attn_out))
        h_attn_out = gate_val * h_attn_out

        h = h_attn_out.view(-1, self.out_channels)

        # 应用dropout
        h = F.dropout(h, self.dropout, training=self.training)

        # 输出投影
        h = self.O_h(h)

        # 残差连接和规范化
        if self.residual:
            h = h_in1 + h  # 残差连接

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # 第二个残差连接

        # 使用改进的FFN
        h = self.FFN_h(h)

        if self.residual:
            h = h_in2 + h  # 残差连接

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual
        )
