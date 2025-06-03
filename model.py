import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    

class AttentionDTI(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000, drug_MAX_LENGH=150, weight_CE=None):
        super(AttentionDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.protein_kernel = hp.protein_kernel

        self.drug_embed = nn.Embedding(65, 160, padding_idx=0)
        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(
            self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH)
        
        # 自注意力增强 - 药物和蛋白质的自注意力层
        self.drug_self_query = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_self_key = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_self_value = nn.Linear(self.conv * 4, self.conv * 4)
        
        self.protein_self_query = nn.Linear(self.conv * 4, self.conv * 4)
        self.protein_self_key = nn.Linear(self.conv * 4, self.conv * 4)
        self.protein_self_value = nn.Linear(self.conv * 4, self.conv * 4)
        
        # 缩放点积注意力 - 交叉注意力层
        self.drug_cross_query = nn.Linear(self.conv * 4, self.conv * 4)
        self.protein_cross_key = nn.Linear(self.conv * 4, self.conv * 4)
        self.protein_cross_value = nn.Linear(self.conv * 4, self.conv * 4)
        
        self.protein_cross_query = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_cross_key = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_cross_value = nn.Linear(self.conv * 4, self.conv * 4)
        
        # 添加注意力温度参数
        self.drug_self_temperature = nn.Parameter(torch.tensor(1.0))
        self.protein_self_temperature = nn.Parameter(torch.tensor(1.0))
        self.drug_cross_temperature = nn.Parameter(torch.tensor(1.0))
        self.protein_cross_temperature = nn.Parameter(torch.tensor(1.0))
        
        # 添加层归一化
        self.drug_self_norm = nn.LayerNorm(self.conv * 4)
        self.protein_self_norm = nn.LayerNorm(self.conv * 4)
        self.drug_cross_norm = nn.LayerNorm(self.conv * 4)
        self.protein_cross_norm = nn.LayerNorm(self.conv * 4)
        
        # 添加前馈神经网络层
        self.drug_ffn = nn.Sequential(
            nn.Linear(self.conv * 4, self.conv * 8),
            nn.GELU(),
            nn.Linear(self.conv * 8, self.conv * 4),
        )
        
        self.protein_ffn = nn.Sequential(
            nn.Linear(self.conv * 4, self.conv * 8),
            nn.GELU(),
            nn.Linear(self.conv * 8, self.conv * 4),
        )
        
        # FFN后的层归一化
        self.drug_ffn_norm = nn.LayerNorm(self.conv * 4)
        self.protein_ffn_norm = nn.LayerNorm(self.conv * 4)
        
        
        # 原有注意力层保留用于兼容
        self.attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.protein_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_CE)

    def scaled_dot_product_attention(self, q, k, v, mask=None, temperature=None):
        """实现带温度参数的缩放点积注意力"""
        d_k = q.size(-1)
        
        # 计算原始点积注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用温度参数调节注意力分布的锐度
        if temperature is not None:
            # 使用clamp限制温度的范围，避免数值不稳定
            temp = torch.clamp(temperature, min=0.1, max=10.0)
            scores = scores / temp
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
        
    def self_attention(self, x, query, key, value, mask=None, temperature=None):
        """实现带温度参数的自注意力"""
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # 线性变换
        q = query(x)
        k = key(x)
        v = value(x)
        
        # 缩放点积注意力
        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask, temperature)
        
        return attn_output

    def forward(self, drug, num_atoms, protein, labels=None):

        drug_masks = []
        for m in range(len(num_atoms)):
            drug_mask = torch.cat([torch.ones(num_atoms[m].long(), dtype=torch.long, device=num_atoms.device),
                                   torch.zeros(150 - num_atoms[m].long(), dtype=torch.long, device=num_atoms.device)], dim=0)
            drug_masks.append(drug_mask)

        drugConv = drug.permute(0, 2, 1)

        proteinembed = self.protein_embed(protein)
        proteinembed = proteinembed.permute(0, 2, 1)
        proteinConv = self.Protein_CNNs(proteinembed)
        
        # 准备mask
        drug_masks = torch.stack(drug_masks, dim=0)
        drug_padding_mask = torch.where(drug_masks > 0, True, False)
        drug_attention_mask = torch.unsqueeze(drug_padding_mask, 2).repeat(1, 1, drugConv.shape[1])
        
        protein_padding_mask = torch.where(protein > 0, True, False)[:, 0:979]
        protein_attention_mask = torch.unsqueeze(protein_padding_mask, 2).repeat(1, 1, proteinConv.shape[1])
        
        # 将特征转换为序列格式 (batch_size, seq_len, hidden_dim)
        drug_features = drugConv.permute(0, 2, 1)
        protein_features = proteinConv.permute(0, 2, 1)
        
        # 1. 自注意力增强
        # 对药物特征应用自注意力
        drug_self_attn = self.self_attention(
            drug_features, 
            self.drug_self_query, 
            self.drug_self_key, 
            self.drug_self_value,
            drug_padding_mask.unsqueeze(1),
            self.drug_self_temperature  # 使用药物自注意力温度参数
        )
        
        # 应用层归一化 (自注意力后)
        drug_self_attn = self.drug_self_norm(drug_self_attn + drug_features)  # 残差连接 + 层归一化
        
        # 对药物特征应用FFN
        drug_ffn_output = self.drug_ffn(drug_self_attn)
        drug_self_final = self.drug_ffn_norm(drug_ffn_output + drug_self_attn)  # 残差连接 + 层归一化
        
        # 对蛋白质特征应用自注意力
        protein_self_attn = self.self_attention(
            protein_features,
            self.protein_self_query,
            self.protein_self_key,
            self.protein_self_value,
            protein_padding_mask.unsqueeze(1),
            self.protein_self_temperature  # 使用蛋白质自注意力温度参数
        )
        
        # 应用层归一化 (自注意力后)
        protein_self_attn = self.protein_self_norm(protein_self_attn + protein_features)  # 残差连接 + 层归一化
        
        # 对蛋白质特征应用FFN
        protein_ffn_output = self.protein_ffn(protein_self_attn)
        protein_self_final = self.protein_ffn_norm(protein_ffn_output + protein_self_attn)  # 残差连接 + 层归一化
        
        # 2. 缩放点积交叉注意力
        # 药物关注蛋白质
        drug_query = self.drug_cross_query(drug_self_final)  # 使用FFN后的特征
        protein_key = self.protein_cross_key(protein_self_final)  # 使用FFN后的特征
        protein_value = self.protein_cross_value(protein_self_final)  # 使用FFN后的特征
        
        # 计算药物对蛋白质的注意力
        drug_protein_attn, _ = self.scaled_dot_product_attention(
            drug_query, protein_key, protein_value, 
            temperature=self.drug_cross_temperature  # 使用药物交叉注意力温度参数
        )
        
        # 应用层归一化 (交叉注意力后)
        drug_cross_attn = self.drug_cross_norm(drug_protein_attn+drug_self_final)  # 残差连接 + 层归一化
        
        # 蛋白质关注药物
        protein_query = self.protein_cross_query(protein_self_final)  # 使用FFN后的特征
        drug_key = self.drug_cross_key(drug_self_final)  # 使用FFN后的特征
        drug_value = self.drug_cross_value(drug_self_final)  # 使用FFN后的特征
        
        # 计算蛋白质对药物的注意力
        protein_drug_attn, _ = self.scaled_dot_product_attention(
            protein_query, drug_key, drug_value,
            temperature=self.protein_cross_temperature  # 使用蛋白质交叉注意力温度参数
        )
        
        # 应用层归一化 (交叉注意力后)
        protein_cross_attn = self.protein_cross_norm(protein_drug_attn+protein_self_final)  # 残差连接 + 层归一化
        
        # 调整维度并进行特征融合
        drug_final = drug_cross_attn.permute(0, 2, 1)  # 转回(batch, hidden, seq_len)格式
        protein_final = protein_cross_attn.permute(0, 2, 1)
        
        drugConv = drug_final
        proteinConv = protein_final
        
        # 应用最大池化
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        # 特征级联
        pair = torch.cat([drugConv, proteinConv], dim=1)
        
        # 全连接层
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        if labels is not None:
            # 计算交叉熵损失
            ce_loss = self.loss_fn(predict, labels)

            # 计算对比损失
            contrast_loss = self.triplet_semihard_with_dynamic_margin(
                pair, labels)

            # 计算L2损失
            l2_loss = 0.0
            # 对不同类型的参数使用不同的正则化系数
            for name, param in self.named_parameters():
                if 'bias' in name:
                    # 偏置项使用较小的正则化系数
                    l2_loss += 0.00001 * torch.norm(param, p=2) ** 2
                else:
                    l2_loss += 0.001 * torch.norm(param, p=2) ** 2


            # 合并三种损失
            total_loss = ce_loss

            return predict, total_loss

        return predict

    def compute_loss(self, predictions, labels):
        """计算预测值和真实标签之间的损失（仅交叉熵损失）"""
        return self.loss_fn(predictions, labels)

    def triplet_semihard_with_dynamic_margin(self, embeddings, labels, base_margin=0.3, margin_factor=1.2, distance='cosine'):
        """
        改进的半硬三元组损失函数，使用动态边界

        Args:
            embeddings: 形状为[batch_size, embedding_dim]的嵌入向量
            labels: 形状为[batch_size]的样本标签
            base_margin: 基础边界值
            margin_factor: 难样本的边界系数(>1.0)
            distance: 距离度量方式，'euclidean'或'cosine'

        Returns:
            loss: 计算得到的改进半硬三元组损失
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # 应用L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算嵌入向量之间的距离矩阵
        if distance == 'euclidean':
            dist_mat = torch.cdist(embeddings, embeddings, p=2).pow(2)
        elif distance == 'cosine':
            # 计算余弦距离 (1 - 相似度)
            cos_sim = torch.matmul(embeddings, embeddings.t())
            dist_mat = 1.0 - cos_sim
        else:
            raise ValueError(f"不支持的距离度量: {distance}")

        # 为每个样本创建正样本和负样本的掩码
        labels = labels.view(-1, 1)
        same_identity_mask = (labels == labels.t()).float()
        diff_identity_mask = (labels != labels.t()).float()

        # 排除对角线(自身)
        eye = torch.eye(batch_size, device=device)
        same_identity_mask = same_identity_mask - eye

        # 计算正样本距离(对每个锚点取平均)
        mask_pos = same_identity_mask.clone()
        mask_pos[mask_pos == 0] = 1e-12  # 避免除零错误
        pos_dist = (dist_mat * same_identity_mask).sum(dim=1) / \
            mask_pos.sum(dim=1)

        # 计算每个负样本的距离和对应的半硬条件掩码
        # 条件: pos_dist < neg_dist < pos_dist + margin
        pos_expanded = pos_dist.unsqueeze(1).expand(batch_size, batch_size)

        # 动态调整边界
        # 首先找出困难的样本: 正样本距离大的锚点
        difficulty = pos_dist / pos_dist.mean()  # 相对困难度
        dynamic_margins = base_margin * \
            torch.clamp(difficulty, min=1.0, max=margin_factor)
        dynamic_margins_expanded = dynamic_margins.unsqueeze(
            1).expand(batch_size, batch_size)

        # 创建半硬三元组掩码
        semihard_mask = (dist_mat > pos_expanded) & (
            dist_mat < pos_expanded + dynamic_margins_expanded) & diff_identity_mask.bool()

        # 计算所有可能的三元组损失
        triplet_loss = torch.tensor(0.0, device=device)
        valid_triplets = 0

        # 对每个锚点处理
        for i in range(batch_size):
            # 获取当前锚点的半硬负样本
            semihard_neg_indices = semihard_mask[i].nonzero(as_tuple=True)[0]

            # 如果存在半硬负样本
            if len(semihard_neg_indices) > 0:
                # 从半硬负样本中选择最近的(最难的)负样本
                closest_negative_dist = dist_mat[i, semihard_neg_indices].min()

                # 计算损失: max(0, pos_dist - neg_dist + margin)
                # 使用当前锚点的动态边界
                curr_loss = F.relu(
                    pos_dist[i] - closest_negative_dist + dynamic_margins[i])
                triplet_loss += curr_loss
                if curr_loss > 0:
                    valid_triplets += 1
            else:
                # 如果没有半硬负样本，退化为使用最近的负样本
                negative_dist = dist_mat[i] * diff_identity_mask[i]
                negative_dist[negative_dist == 0] = 1e9  # 将非负样本距离设为极大值
                closest_negative_dist = negative_dist.min()

                # 计算损失
                curr_loss = F.relu(
                    pos_dist[i] - closest_negative_dist + dynamic_margins[i] * margin_factor)  # 增加边界值
                triplet_loss += curr_loss
                if curr_loss > 0:
                    valid_triplets += 1

        # 如果存在有效的三元组，则返回平均损失
        if valid_triplets > 0:
            return triplet_loss / valid_triplets
        else:
            return torch.tensor(0.0, device=device)
