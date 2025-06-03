import torch
import argparse
import os
from model import AttentionDTI
from proteins import CHARPROTSET, collate_fn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 定义超参数类，与原始代码保持一致


class HyperParameters:
    def __init__(self):
        self.char_dim = 128  # 字符嵌入维度
        self.conv = 40       # CNN卷积通道数
        self.protein_kernel = [4, 8, 12]  # 蛋白质CNN核大小
        self.use_protbert = True          # 是否使用ProtBERT
        self.protbert_finetune = False    # 是否微调ProtBERT参数

# 简单数据集示例


class SimpleProteinDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels if labels is not None else [0] * len(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # 将序列转换为整数索引
        seq_encoded = [CHARPROTSET.get(aa, 0) for aa in seq]
        seq_encoded = torch.tensor(seq_encoded, dtype=torch.long)

        # 填充到固定长度
        padded_seq = torch.zeros(1000, dtype=torch.long)
        padded_seq[:len(seq_encoded)] = seq_encoded

        return padded_seq, torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description='ProtBERT处理蛋白质序列示例')
    parser.add_argument(
        '--use_protbert', action='store_true', help='是否使用ProtBERT')
    parser.add_argument('--finetune', action='store_true', help='是否微调ProtBERT')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化超参数
    hp = HyperParameters()
    hp.use_protbert = args.use_protbert
    hp.protbert_finetune = args.finetune

    # 创建模型
    model = AttentionDTI(hp)
    model = model.to(device)

    # 可以通过此方法动态切换ProtBERT模式
    # model.set_protbert_mode(use_protbert=True, finetune=False)

    print(f"使用ProtBERT: {model.use_protbert}")
    print(f"微调ProtBERT: {model.protbert_finetune}")

    # 简单的蛋白质序列示例
    sample_proteins = [
        "MKKIFFVLTLLFSSSLAYGQNMEQFVQSVKVGDKVTLNCDYSSSSDSSYVYWYRQRPGQGPEFIVYYIRASSIYSTDKFNNGVRMASSLGKKDAASITSAIQLANKASDGIMIQTPPSLSVPNERGDSVTLMCQVSGDFPKDYTALTWWKDGQKLAEVKRRSVETSESPAKPTLHPISADPED",
        "MVSATQQHSRTQEVLGVGTVIIWVTFMATLAGLTLPVNLSWSHLGNEATKVLLYGLFLASVLGGLVFALWCVRMRRHHCRVSDMPLSA",
        "MVLSQMLLFSSLVLLYLSQVSAQEVTVNCTYTNNKDVSTTWKALQNGSEMTFRCQGDGSWGPDLRWMLPVREGVAQKVLTFDSGTQSGNQRLTCAARGSWFHNPRLVSRTLKDNKDRKILLVPTQEVPTAEVQETSPPEALREGEDAVFTCESGAFSMPWYLNYHSDRFHLQLNTTDRTQLSKAGLLTLYSIRLEKED",
    ]

    # 创建简单数据集和数据加载器
    dataset = SimpleProteinDataset(sample_proteins)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # 评估模式
    model.eval()

    # 假设药物输入
    # [batch_size, drug_MAX_LENGH, embedding_dim]
    dummy_drug = torch.zeros(2, 150, 160).to(device)
    dummy_num_atoms = torch.tensor([50, 60]).to(device)  # 假设的原子数量

    # 前向传播示例
    with torch.no_grad():
        for proteins, labels in tqdm(dataloader, desc="处理蛋白质"):
            proteins = proteins.to(device)
            labels = labels.to(device)

            # 模型推理
            outputs = model(dummy_drug, dummy_num_atoms, proteins)
            probabilities = F.softmax(outputs, dim=1)

            print("\n预测结果:")
            for i, prob in enumerate(probabilities):
                print(
                    f"蛋白质 {i+1}: 类别0概率 = {prob[0]:.4f}, 类别1概率 = {prob[1]:.4f}")

    print("\n示例完成!")


if __name__ == "__main__":
    main()
