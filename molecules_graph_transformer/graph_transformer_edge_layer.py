import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

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


# 标准化的softmax注意力计算
def softmax_attention(field):
    def func(edges):
        # 数值稳定性裁剪
        attention = edges.data[field].clamp(-5, 5)
        # 在最后一个维度上应用softmax
        attention = F.softmax(attention, dim=-1)
        return {field: attention}
    return func


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # 添加注意力温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 添加注意力dropout
        self.attn_dropout = nn.Dropout(0.1)

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.pos_encoder = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.pos_encoder = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # 添加层归一化
        self.layer_norm = nn.LayerNorm(out_dim * num_heads)
        
        self.pos_embedding = nn.Parameter(torch.FloatTensor(1, out_dim * num_heads))
        nn.init.xavier_uniform_(self.pos_embedding)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))

        # 使用温度参数缩放
        g.apply_edges(scaling('score', np.sqrt(self.out_dim) * self.temperature))

        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))

        # Add spatial position with learnable encoding
        g.apply_edges(add_spatial_pos('score', 'learnable_pos'))

        # 应用注意力dropout
        g.edata['score'] = self.attn_dropout(g.edata['score'])

        # 使用标准化的softmax注意力计算
        g.apply_edges(softmax_attention('score'))

        # 复制边特征作为e_out
        g.apply_edges(out_edge_features('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e, spatial_pos):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # 应用层归一化
        Q_h = self.layer_norm(Q_h)
        K_h = self.layer_norm(K_h)
        V_h = self.layer_norm(V_h)

        # 使用可学习的投影
        proj_e = self.proj_e(e)
        proj_e = self.layer_norm(proj_e)

        # 将空间位置信息与可学习的位置编码结合
        learnable_pos = self.pos_encoder(spatial_pos) + self.pos_embedding
        learnable_pos = self.layer_norm(learnable_pos)

        # Reshaping into [num_nodes, num_heads, feat_dim]
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
        g.edata['learnable_pos'] = learnable_pos.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        # 对分母添加epsilon以确保数值稳定性
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        
        # 确保e_out存在
        if 'e_out' not in g.edata:
            g.edata['e_out'] = g.edata['score']
        e_out = g.edata['e_out']

        return h_out, e_out


class GraphTransformerLayer(nn.Module):
    """
        Graph Transformer Layer with improved attention mechanism
        and feed-forward networks for both nodes and edges
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

        self.attention = MultiHeadAttentionLayer(
            in_dim, out_dim//num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        # 维度不匹配时的投影层
        self.dim_match = (in_dim == out_dim)
        if not self.dim_match:
            self.proj_h = nn.Linear(in_dim, out_dim)
            self.proj_e = nn.Linear(in_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(in_dim)
            self.layer_norm1_e = nn.LayerNorm(in_dim)
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

        # 前馈网络
        self.FFN_h = nn.Sequential(
            nn.Linear(out_dim, out_dim*4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(out_dim*4, out_dim)
        )

        self.FFN_e = nn.Sequential(
            nn.Linear(out_dim, out_dim*4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(out_dim*4, out_dim)
        )

    def forward(self, g, h, e, spatial_pos):
        # 第一个子层：多头注意力
        h_in1 = h
        e_in1 = e

        # PreLN - 先应用层归一化
        if self.layer_norm:
            h_norm = self.layer_norm1_h(h_in1)
            e_norm = self.layer_norm1_e(e_in1)
        else:
            h_norm = h_in1
            e_norm = e_in1

        # 注意力机制
        h_attn_out, e_attn_out = self.attention(g, h_norm, e_norm, spatial_pos)

        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        # 残差连接 - 处理维度不匹配情况
        if self.residual:
            if self.dim_match:
                h = h_in1 + h
                e = e_in1 + e
            else:
                h = self.proj_h(h_in1) + h
                e = self.proj_e(e_in1) + e

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        # 第二个子层：前馈网络
        h_in2 = h
        e_in2 = e

        # PreLN - 先应用层归一化
        if self.layer_norm:
            h_norm = self.layer_norm2_h(h_in2)
            e_norm = self.layer_norm2_e(e_in2)
        else:
            h_norm = h_in2
            e_norm = e_in2

        # 前馈网络
        h = self.FFN_h(h_norm)
        e = self.FFN_e(e_norm)

        # 残差连接
        if self.residual:
            h = h_in2 + h
            e = e_in2 + e

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads, self.residual)
