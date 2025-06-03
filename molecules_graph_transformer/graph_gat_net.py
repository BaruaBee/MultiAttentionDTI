import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np

class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 消息传递网络
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 修改输入维度
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 更新网络
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, g, h, e):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['e'] = e
            
            # 计算边特征
            g.apply_edges(self.edge_message)
            
            # 消息聚合
            g.update_all(self.message_func, self.reduce_func)
            
            # 更新节点特征
            h_new = self.update_mlp(torch.cat([h, g.ndata['agg_msg']], dim=-1))
            
            return h_new, g.edata['e_new']

    def edge_message(self, edges):
        # 构建消息 - 使用边特征和位置信息
        msg = torch.cat([
            edges.src['h'], 
            edges.dst['h'], 
            edges.data['e']
        ], dim=-1)
        msg = self.msg_mlp(msg)
        
        # 更新边特征
        e_new = edges.data['e'] + msg
        
        return {'msg': msg, 'e_new': e_new}

    def message_func(self, edges):
        # 加入边特征权重
        weight = torch.sigmoid(self.edge_weight(edges.data['e_new']))
        return {'msg': edges.data['msg'] * weight}

    def reduce_func(self, nodes):
        # 考虑邻居数量
        num_neighbors = nodes.mailbox['msg'].shape[1]
        return {'agg_msg': torch.sum(nodes.mailbox['msg'], dim=1) / (num_neighbors + 1e-6)}


class GraphEGNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params.num_atom_type
        num_bond_type = net_params.num_bond_type
        num_in_degree = net_params.num_in_degree
        num_out_degree = net_params.num_out_degree
        num_spatial_pos = net_params.num_spatial_pos
        self.hidden_dim = net_params.hidden_dim
        num_heads = net_params.n_heads
        out_dim = net_params.out_dim
        in_feat_dropout = net_params.in_feat_dropout
        dropout = net_params.dropout
        n_layers = net_params.L
        self.layer_norm = net_params.layer_norm
        self.batch_norm = net_params.batch_norm
        self.residual = net_params.residual
        
        # 输入嵌入层
        self.embedding_h = nn.Embedding(num_atom_type, self.hidden_dim)
        self.embedding_in_degree = nn.Embedding(num_in_degree, self.hidden_dim)
        self.embedding_out_degree = nn.Embedding(num_out_degree, self.hidden_dim)
        self.embedding_spatial_pos = nn.Embedding(num_spatial_pos, self.hidden_dim)
        self.embedding_e = nn.Embedding(num_bond_type, self.hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # EGNN层
        self.egnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.egnn_layers.append(EGNNLayer(
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ))
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, out_dim)
        )
        
        # 层归一化
        if self.layer_norm:
            self.node_norm = nn.LayerNorm(self.hidden_dim)
            self.edge_norm = nn.LayerNorm(self.hidden_dim)
        
        # 批归一化
        if self.batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(self.hidden_dim) for _ in range(n_layers)
            ])
        
        # 边特征权重网络
        self.edge_weight = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # 边特征更新网络
        self.edge_update = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, g, h, e, in_degree, out_degree, spatial_pos):
        # 输入嵌入
        h = self.embedding_h(h)
        h = h + self.embedding_in_degree(in_degree) + self.embedding_out_degree(out_degree)
        h = self.in_feat_dropout(h)
        
        # 边特征处理 - 加入位置信息
        e = self.embedding_e(e)
        pos = self.embedding_spatial_pos(spatial_pos)
        e = e + pos  # 将位置信息融入边特征
        
        if self.layer_norm:
            e = self.edge_norm(e)
        
        # EGNN层处理
        for i, egnn_layer in enumerate(self.egnn_layers):
            h_in = h
            e_in = e
            
            # 应用EGNN层
            h, e = egnn_layer(g, h, e)
            
            # 残差连接
            if self.residual:
                h = h_in + h
                e = e_in + e
            
            # 边特征更新
            e = self.edge_update(torch.cat([e, pos], dim=-1))
            
            # 层归一化
            if self.layer_norm:
                h = self.node_norm(h)
                e = self.edge_norm(e)
            
            # 批归一化
            if self.batch_norm:
                h = self.batch_norm_layers[i](h)
            
            # 激活函数和dropout
            h = F.dropout(h, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)
        
        # 输出层
        h = self.output_layer(h)
        
        return h 