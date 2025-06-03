import numpy as np
import torch
from datetime import datetime
import os


class EarlyStopping:
    """基于验证集AUC的早停机制，当AUC在指定耐心期内未改善时提前停止训练。"""

    def __init__(self, savepath=None, patience=7, verbose=False, delta=0, num_n_fold=0):
        """
        Args:
            patience (int): 自上次验证AUC改善后等待的轮数。
                            默认: 7
            verbose (bool): 如果为True，每次验证AUC改善时会打印消息。
                            默认: False
            delta (float): 被视为改善的最小变化量。
                            默认: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # 记录验证AUC历史
        self.auc_history = []
        # 记录最佳模型的epoch
        self.best_epoch = 0

        # 创建日志文件
        self.log_file = os.path.join(savepath, 'early_stopping_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Early Stopping Configuration:\n")
            f.write(f"Patience: {patience}\n")
            f.write(f"Delta: {delta}\n")
            f.write(f"Start Time: {self.current_time}\n")
            f.write("-" * 50 + "\n")

    def __call__(self, AUC_dev, model_drug, model, num_epoch):
        # 记录每个epoch的验证AUC
        self.auc_history.append(AUC_dev)

        score = AUC_dev  # AUC越高越好，不需要取负值

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = num_epoch
            self.save_checkpoint(AUC_dev, model_drug, model, num_epoch)
            self._log_status(num_epoch, AUC_dev, "初始化最佳模型")
        elif score <= self.best_score + self.delta:
            self.counter += 1
            message = f'EarlyStopping counter: {self.counter} out of {self.patience}'
            print(message)
            self._log_status(num_epoch, AUC_dev,
                             f"未改进 ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered after epoch {num_epoch}")
                self._log_status(num_epoch, AUC_dev, "触发早停! 训练结束")
                return True
        else:
            self.best_score = score
            self.best_epoch = num_epoch
            self.save_checkpoint(AUC_dev, model_drug, model, num_epoch)
            self.counter = 0
            self._log_status(num_epoch, AUC_dev, "模型改进，重置计数器")

        return self.early_stop

    def save_checkpoint(self, AUC_dev, model_drug, model, num_epoch):
        '''当验证AUC提高时保存模型。'''
        if self.verbose:
            print(
                f'验证AUC提高 ({self.val_loss_min:.6f} --> {AUC_dev:.6f})。保存模型...')

        # 先保存为临时文件，再重命名，防止保存过程中出错导致模型损坏
        drug_path = os.path.join(
            self.savepath, 'model_drug_best_checkpoint_temp.pth')
        model_path = os.path.join(
            self.savepath, 'model_best_checkpoint_temp.pth')

        torch.save(model_drug, drug_path)
        torch.save(model, model_path)

        # 如果保存成功，再重命名为最终文件
        final_drug_path = os.path.join(
            self.savepath, 'model_drug_best_checkpoint.pth')
        final_model_path = os.path.join(
            self.savepath, 'model_best_checkpoint.pth')

        if os.path.exists(drug_path) and os.path.exists(model_path):
            # 如果已存在最终文件，先删除
            if os.path.exists(final_drug_path):
                os.remove(final_drug_path)
            if os.path.exists(final_model_path):
                os.remove(final_model_path)

            os.rename(drug_path, final_drug_path)
            os.rename(model_path, final_model_path)

            # 记录最佳epoch和验证AUC
            with open(os.path.join(self.savepath, 'best_model_info.txt'), 'w') as f:
                f.write(f"Best Epoch: {self.best_epoch}\n")
                f.write(f"Best Validation AUC: {AUC_dev:.6f}\n")
                f.write(
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.val_loss_min = AUC_dev

    def _log_status(self, epoch, AUC_dev, status):
        """记录早停状态到日志文件"""
        with open(self.log_file, 'a') as f:
            f.write(
                f"Epoch {epoch}: AUC_dev={AUC_dev:.6f}, status={status}\n")

    def get_best_epoch(self):
        """返回最佳模型的epoch"""
        return self.best_epoch

class DualEarlyStopping:
    """基于验证集AUC和验证损失的双重早停机制，当两个条件都满足时才停止训练。"""

    def __init__(self, savepath=None, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 自上次验证指标改善后等待的轮数。
            verbose (bool): 如果为True，每次验证指标改善时会打印消息。
            delta (float): 被视为改善的最小变化量。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter_auc = 0
        self.counter_loss = 0
        self.best_auc = None
        self.best_loss = None
        self.early_stop_auc = False
        self.early_stop_loss = False
        self.delta = delta
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        # 记录验证指标历史
        self.auc_history = []
        self.loss_history = []
        # 记录最佳模型的epoch
        self.best_auc_epoch = 0
        self.best_loss_epoch = 0

        # 创建日志文件
        self.log_file = os.path.join(savepath, 'dual_early_stopping_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Dual Early Stopping Configuration:\n")
            f.write(f"Patience: {patience}\n")
            f.write(f"Delta: {delta}\n")
            f.write(f"Start Time: {datetime.now().strftime('%b%d_%H-%M-%S')}\n")
            f.write("-" * 50 + "\n")

    def __call__(self, AUC_dev, valid_loss, model_drug, model, num_epoch):
        # 记录每个epoch的验证指标
        self.auc_history.append(AUC_dev)
        self.loss_history.append(valid_loss)

        # 检查AUC早停
        if self.best_auc is None:
            self.best_auc = AUC_dev
            self.best_auc_epoch = num_epoch
            self.save_checkpoint('auc', AUC_dev, model_drug, model, num_epoch)
            print(f'[Epoch {num_epoch}] 初始AUC: {AUC_dev:.6f}')
            self._log_status(num_epoch, AUC_dev, valid_loss, "初始化最佳AUC模型")
        elif AUC_dev <= self.best_auc + self.delta:  # AUC没有提高
            self.counter_auc += 1
            print(f'[Epoch {num_epoch}] AUC未提高 ({self.counter_auc}/{self.patience})')
            self._log_status(num_epoch, AUC_dev, valid_loss, f"AUC未改进 ({self.counter_auc}/{self.patience})")
            if self.counter_auc >= self.patience:
                self.early_stop_auc = True
        else:  # AUC提高了
            print(f'[Epoch {num_epoch}] AUC提高 ({self.best_auc:.6f} --> {AUC_dev:.6f})')
            self.best_auc = AUC_dev
            self.best_auc_epoch = num_epoch
            self.save_checkpoint('auc', AUC_dev, model_drug, model, num_epoch)
            self.counter_auc = 0
            self._log_status(num_epoch, AUC_dev, valid_loss, "AUC模型改进，重置计数器")

        # 检查损失早停
        if self.best_loss is None:
            self.best_loss = valid_loss
            self.best_loss_epoch = num_epoch
            self.save_checkpoint('loss', valid_loss, model_drug, model, num_epoch)
            print(f'[Epoch {num_epoch}] 初始损失: {valid_loss:.6f}')
            self._log_status(num_epoch, AUC_dev, valid_loss, "初始化最佳损失模型")
        elif valid_loss >= self.best_loss - self.delta:  # 损失没有降低
            self.counter_loss += 1
            print(f'[Epoch {num_epoch}] 损失未降低 ({self.counter_loss}/{self.patience})')
            self._log_status(num_epoch, AUC_dev, valid_loss, f"损失未改进 ({self.counter_loss}/{self.patience})")
            if self.counter_loss >= self.patience:
                self.early_stop_loss = True
        else:  # 损失降低了
            print(f'[Epoch {num_epoch}] 损失降低 ({self.best_loss:.6f} --> {valid_loss:.6f})')
            self.best_loss = valid_loss
            self.best_loss_epoch = num_epoch
            self.save_checkpoint('loss', valid_loss, model_drug, model, num_epoch)
            self.counter_loss = 0
            self._log_status(num_epoch, AUC_dev, valid_loss, "损失模型改进，重置计数器")

        # 只有当两个早停都触发时才返回True
        if self.early_stop_auc and self.early_stop_loss:
            print(f"[Epoch {num_epoch}] 双重早停已触发，训练结束")
            self._log_status(num_epoch, AUC_dev, valid_loss, "双重早停触发! 训练结束")
            return True
        return False

    def save_checkpoint(self, metric_type, metric_value, model_drug, model, num_epoch):
        '''当验证指标提高时保存模型。'''
        if self.verbose:
            print(f'验证{metric_type}提高，保存模型...')

        # 先保存为临时文件，再重命名，防止保存过程中出错导致模型损坏
        drug_path = os.path.join(self.savepath, f'model_drug_best_{metric_type}_temp.pth')
        model_path = os.path.join(self.savepath, f'model_best_{metric_type}_temp.pth')

        torch.save(model_drug, drug_path)
        torch.save(model, model_path)

        # 如果保存成功，再重命名为最终文件
        final_drug_path = os.path.join(self.savepath, f'model_drug_best_{metric_type}.pth')
        final_model_path = os.path.join(self.savepath, f'model_best_{metric_type}.pth')

        if os.path.exists(drug_path) and os.path.exists(model_path):
            # 如果已存在最终文件，先删除
            if os.path.exists(final_drug_path):
                os.remove(final_drug_path)
            if os.path.exists(final_model_path):
                os.remove(final_model_path)

            os.rename(drug_path, final_drug_path)
            os.rename(model_path, final_model_path)

            # 记录最佳epoch和验证指标
            with open(os.path.join(self.savepath, f'best_{metric_type}_model_info.txt'), 'w') as f:
                f.write(f"Best Epoch: {num_epoch}\n")
                f.write(f"Best Validation {metric_type}: {metric_value:.6f}\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _log_status(self, epoch, AUC_dev, valid_loss, status):
        """记录早停状态到日志文件"""
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}: AUC_dev={AUC_dev:.6f}, valid_loss={valid_loss:.6f}, status={status}\n")

    def get_best_epochs(self):
        """返回两个最佳模型的epoch"""
        return self.best_auc_epoch, self.best_loss_epoch
