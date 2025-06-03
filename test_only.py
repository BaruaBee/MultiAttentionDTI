import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pickle
import sys
from tqdm import tqdm
from torch.serialization import add_safe_globals
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc

# 导入所有需要的类
from model import AttentionDTI
from proteins import CustomDataSet, collate_fn

# 确保正确导入所有分子相关类，关键是确保导入的顺序与原始序列化时相同
from molecules import MoleculeDataset, MoleculeDatasetDGL, MoleculeDGL
from molecules_graph_transformer.graph_transformer_net import GraphTransformerNet
import molecules  # 导入整个模块，确保所有类都可用

# 将包含分子类定义的模块添加到pickle能识别的模块中
sys.modules['molecules'] = molecules

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def test_precess(model, model_drug, test_loader, drug_load, LOSS):
    model.eval()
    model_drug.eval()
    test_losses = []
    Y, P, S = [], [], []
    dataloader_iterator = iter(drug_load)

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                break

            batch_graphs, num_atoms, in_degrees, out_degrees, spatial_pos, edge_inputs = batch
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat']  # num x feat
            edge_inputs = edge_inputs.to(device)
            num_atoms = num_atoms.to(device)
            in_degrees = in_degrees.to(device)
            out_degrees = out_degrees.to(device)
            spatial_pos = spatial_pos.to(device)

            compounds = model_drug.forward(
                batch_graphs, batch_x, edge_inputs, in_degrees, out_degrees, spatial_pos)

            y = [x for x in compounds.split(list(num_atoms), dim=0)]
            drug_embed = nn.Embedding(65, 160, padding_idx=0, device=device)
            outt = []
            for m in range(len(num_atoms)):
                drugem = torch.zeros(
                    150 - num_atoms[m].long(), dtype=torch.long, device=device)
                drugembed = drug_embed(drugem)
                out = torch.cat([y[m], drugembed], dim=0)
                outt.append(out)
            compounds = torch.stack(outt, dim=0)

            proteins, labels = data
            proteins = proteins.to(device)
            labels = labels.to(device)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            predicted_scores = model(compounds, num_atoms, proteins)

            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(
                predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())

    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)

    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC


def main():
    # 设置数据集名称
    DATASET = "Davis"  # 可以修改为您需要的数据集

    # 设置模型路径
    save_path = f"./{DATASET}_drug_inductive_setting"
    best_model_path = os.path.join(save_path, 'model_best_checkpoint.pth')
    best_model_drug_path = os.path.join(
        save_path, 'model_drug_best_checkpoint.pth')

    # 加载测试数据
    print("加载测试数据集...")

    # 从文件加载数据
    dir_input = f'./protein_data/{DATASET}.txt'
    with open(dir_input, "r") as f:
        dataset = f.read().strip().split('\n')

    # 我们需要像原始脚本一样分割测试数据
    # 提取药物ID和蛋白质ID
    drug_ids, protein_ids = [], []
    for pair in dataset:
        pair = pair.strip().split()
        drug_id, protein_id = pair[-4], pair[-3]
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)
    drug_set = list(set(drug_ids))
    drug_set.sort(key=drug_ids.index)

    # 使用相同的种子分割数据
    SEED = 0
    torch.manual_seed(SEED)
    _, test_d = torch.utils.data.random_split(drug_set, [math.ceil(
        0.5*len(drug_set)), len(drug_set)-math.ceil(0.5*len(drug_set))])
    test_drug = []
    for i in range(len(test_d)):
        pair = drug_set[test_d.indices[i]]
        test_drug.append(pair)

    # 提取测试数据集
    test_dataset = []
    for i in dataset:
        pair = i.strip().split()
        drug_id = pair[-4]
        if drug_id in test_drug:
            test_dataset.append(i)

    # 创建测试数据集
    test_dataset = CustomDataSet(test_dataset)
    test_dataset_load = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 从文件加载药物数据集前，修复pickle反序列化问题
    # 将pickle版本设置为兼容模式
    pickle.HIGHEST_PROTOCOL = 4

    # 加载药物数据集
    # 替代加载方式，手动处理pickle加载
    try:
        drug_dataset = MoleculeDataset(
            f'{DATASET}_drug_inductive_setting_{SEED}')
        test_drug_dataset = drug_dataset.test
    except AttributeError as e:
        print(f"错误加载药物数据集: {e}")
        print("尝试使用替代方法加载...")
        # 尝试直接加载测试数据集
        test_drug_dataset_path = f'./{DATASET}_drug_inductive_setting/test_drug_dataset.pkl'
        if os.path.exists(test_drug_dataset_path):
            with open(test_drug_dataset_path, 'rb') as f:
                test_drug_dataset = pickle.load(f)
        else:
            print(f"找不到测试药物数据集文件: {test_drug_dataset_path}")
            return

    test_drug_load = DataLoader(test_drug_dataset, batch_size=32, shuffle=False,
                                num_workers=0, collate_fn=drug_dataset.collate)

    print("数据加载完成")

    # 设置损失函数
    # 根据您的数据集调整权重
    if DATASET == "Davis":
        weight_CE = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_CE = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_CE = None

    LOSS = nn.CrossEntropyLoss(weight=weight_CE)

    # 加载模型
    print(f"加载模型: {best_model_path}")

    # 添加安全类列表
    add_safe_globals([AttentionDTI, GraphTransformerNet])

    # 检查模型文件是否存在
    if os.path.exists(best_model_path) and os.path.exists(best_model_drug_path):
        # 使用weights_only=False加载模型
        model = torch.load(best_model_path, weights_only=False)
        model_drug = torch.load(best_model_drug_path, weights_only=False)
        print("模型加载成功")

        # 运行测试
        print("开始测试...")
        Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC = test_precess(
            model, model_drug, test_dataset_load, test_drug_load, LOSS)

        # 打印结果
        print(f"测试结果:")
        print(f"损失: {test_loss:.5f}")
        print(f"准确率: {Accuracy:.5f}")
        print(f"精确率: {Precision:.5f}")
        print(f"召回率: {Recall:.5f}")
        print(f"AUC: {AUC:.5f}")
        print(f"AUPR: {PRC:.5f}")

        # 保存预测结果
        result_file = os.path.join(save_path, f"{DATASET}_test_prediction.txt")
        with open(result_file, 'w') as f:
            for i in range(len(Y)):
                f.write(f"{Y[i]} {P[i]}\n")
        print(f"预测结果已保存到: {result_file}")
    else:
        print(f"错误: 模型文件不存在")


if __name__ == "__main__":
    # 导入math模块，用于数据集分割
    import math
    main()
