import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GINEConv, EGATConv, GATConv

from molecules_graph_transformer.graph_transformer_edge_layer import GraphTransformerLayer
from molecules_graph_transformer.graph_transformer_node_layer import NodeGraphTransformerLayer


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params.num_atom_type
        num_bond_type = net_params.num_bond_type
        num_in_degree = net_params.num_in_degree
        num_out_degree = net_params.num_out_degree
        num_spatial_pos = net_params.num_spatial_pos
        hidden_dim = net_params.hidden_dim
        num_heads = net_params.n_heads
        out_dim = net_params.out_dim
        in_feat_dropout = net_params.in_feat_dropout
        dropout = net_params.dropout
        n_layers = net_params.L
        self.layer_norm = net_params.layer_norm
        self.batch_norm = net_params.batch_norm
        self.residual = net_params.residual

        # 添加损失函数类型和正则化参数
        self.loss_type = net_params.loss_type if hasattr(
            net_params, 'loss_type') else 'l1'
        self.weight_decay = net_params.weight_decay if hasattr(
            net_params, 'weight_decay') else 0.0

        # 获取local和global层的数量
        n_local_layers = n_layers // 2
        # n_local_layers = n_layers // 2
        n_global_layers = n_layers - n_local_layers

        # 保存hidden_dim用于后续使用
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        self.embedding_in_degree = nn.Embedding(num_in_degree, hidden_dim)
        self.embedding_out_degree = nn.Embedding(num_out_degree, hidden_dim)
        self.embedding_spatial_pos = nn.Embedding(num_spatial_pos, hidden_dim)
        self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        self.e_dis_encoder = nn.Embedding(3*hidden_dim*hidden_dim, 1)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # Local层 - 使用EGATConv（边特征图注意力网络）
        self.local_layers = nn.ModuleList()
        for i in range(n_local_layers):
            self.local_layers.append(EGATConv(
                in_node_feats=hidden_dim,
                in_edge_feats=hidden_dim,
                out_node_feats=hidden_dim,
                out_edge_feats=hidden_dim,
                num_heads=num_heads
            ))

        # 多头注意力合并投影层
        self.multihead_node_projs = nn.Linear(
            hidden_dim * num_heads, hidden_dim)
        self.multihead_edge_projs = nn.Linear(
            hidden_dim * num_heads, hidden_dim)
        
        # 添加投影层的正则化
        self.node_proj_norm = nn.LayerNorm(hidden_dim)
        self.edge_proj_norm = nn.LayerNorm(hidden_dim)

        # Global层 - 使用GraphTransformerLayer
        self.global_layers = nn.ModuleList()
        for i in range(n_global_layers-1):
            self.global_layers.append(GraphTransformerLayer(
                hidden_dim,
                hidden_dim,
                num_heads,
                dropout,
                self.layer_norm,
                self.batch_norm,
                self.residual,
                use_bias=True,
            ))
        # 最后一层输出所需维度
        self.global_layers.append(GraphTransformerLayer(
            hidden_dim,
            out_dim,
            num_heads,
            dropout,
            self.layer_norm,
            self.batch_norm,
            self.residual,
            use_bias=True
        ))

        # 处理层间投影和归一化
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        # 添加边投影层
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)

        # 添加投影后的dropout层
        self.node_dropout = nn.Dropout(dropout)
        # 添加边dropout层
        self.edge_dropout = nn.Dropout(dropout)

        # 层归一化
        if self.layer_norm:
            self.layer_norm_local = nn.LayerNorm(hidden_dim)
            # 添加边的层归一化
            self.edge_norm_local = nn.LayerNorm(hidden_dim)

    def forward(self, g, h, e, in_degree, out_degree, spatial_pos):
        # 输入嵌入
        h = self.embedding_h(h)
        h = h + self.embedding_in_degree(in_degree) + \
            self.embedding_out_degree(out_degree)
        h = self.in_feat_dropout(h)

        # 边特征处理
        edge_feat = self.embedding_e(e)
        e_flat = edge_feat.permute(1, 0, 2)
        e_flat = torch.bmm(
            e_flat, self.e_dis_encoder.weight.reshape(-1, self.hidden_dim, self.hidden_dim))
        e = e_flat.sum(0) / (spatial_pos.float().unsqueeze(-1) + 1e-6)

        # 处理空间位置信息
        spatial_emb = self.embedding_spatial_pos(spatial_pos)

        # Local层处理 - EGAT
        h_local = h
        e_local = e
        for i, egat_layer in enumerate(self.local_layers):
            # EGATConv返回节点特征和边特征
            h_out, e_out = egat_layer(g, h_local, e_local)

            # 使用投影层合并多头注意力
            if len(h_out.shape) > 2:  # [N, num_heads, feat_dim]
                batch_size = h_out.size(0)
                h_out = h_out.reshape(batch_size, -1)
                h_out = self.multihead_node_projs(h_out)  # [N, hidden_dim]
                h_out = self.node_proj_norm(h_out)  # 添加层归一化

                edge_size = e_out.size(0)
                e_out = e_out.reshape(edge_size, -1)  # [E, num_heads*feat_dim]
                e_out = self.multihead_edge_projs(e_out)  # [E, hidden_dim]
                e_out = self.edge_proj_norm(e_out)  # 添加层归一化

            h_out = self.node_dropout(h_out)

            e_out = self.edge_dropout(e_out)

            # 残差连接 - 节点
            if self.residual:
                h_local = h_out + h_local
            else:
                h_local = h_out

            # 残差连接 - 边
            if self.residual:
                e_local = e_out + e_local
            else:
                e_local = e_out

        # Global层处理 - GraphTransformerLayer
        h_global = h_local
        e = e_local
        for i, graph_transformer in enumerate(self.global_layers):
            h_global, e = graph_transformer(g, h_global, e, spatial_emb)

        return h_global
