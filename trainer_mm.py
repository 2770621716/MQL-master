"""
多模态联合训练器
适配 MMRQVAE（一个模型包含文本和图像两个 RQ）
"""

import logging
import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from collections import defaultdict
from utils import ensure_dir, set_color
import os
from diagnose_lora import LoRADiagnostics


class MMTrainer(object):
    """多模态训练器 - 适配 MMRQVAE"""

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = torch.device(args.device)
        self.ckpt_dir = args.ckpt_dir
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_text = np.inf
        self.best_collision_image = np.inf

        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        
        # LoRA 诊断工具（暂时关闭）
        # self.diagnostics = LoRADiagnostics(
        #     model=self.model,
        #     output_dir=os.path.join(self.ckpt_dir, "diagnostics")
        # )

    def _build_optimizer(self):
        """构建优化器"""
        params = self.model.parameters()
        
        if self.learner.lower() == 'adamw':
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adam':
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            return optim.Adam(params, lr=self.lr)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        total_align_loss = 0

        pbar = tqdm(train_data, total=len(train_data), ncols=100,
                   desc=set_color(f"Train {epoch_idx}", "pink"))

        for batch_text, batch_image, _ in pbar:
            batch_text = batch_text.to(self.device)
            batch_image = batch_image.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播（硬量化）
            (text_out, image_out, 
             text_rq_loss, image_rq_loss,
             text_indices, image_indices,
             z_q_text, z_q_image) = self.model(batch_text, batch_image)

            # 计算损失
            loss, loss_recon, loss_quant, align_loss = self.model.compute_loss(
                text_out, image_out,
                text_rq_loss, image_rq_loss,
                z_q_text, z_q_image,
                batch_text, batch_image
            )

            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_quant_loss += loss_quant.item()
            total_align_loss += align_loss.item()

        return total_loss, total_recon_loss, total_quant_loss, total_align_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        """评估"""
        self.model.eval()

        indices_list_text = []
        indices_list_image = []
        num_sample = 0

        for batch_text, batch_image, _ in tqdm(valid_data, desc=set_color("Eval", "pink"), leave=False):
            num_sample += len(batch_text)
            batch_text = batch_text.to(self.device)
            batch_image = batch_image.to(self.device)

            text_indices, image_indices = self.model.get_indices(text_x=batch_text, image_x=batch_image)

            indices = text_indices.view(-1, text_indices.shape[-1]).cpu().numpy()
            for index in indices:
                indices_list_text.append("-".join([str(int(_)) for _ in index]))

            indices = image_indices.view(-1, image_indices.shape[-1]).cpu().numpy()
            for index in indices:
                indices_list_image.append("-".join([str(int(_)) for _ in index]))

        collision_text = (num_sample - len(set(indices_list_text))) / num_sample
        collision_image = (num_sample - len(set(indices_list_image))) / num_sample

        return collision_text, collision_image

    def _save_checkpoint(self, epoch, name):
        """保存检查点"""
        state = {
            "args": self.args,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        ckpt_path = os.path.join(self.ckpt_dir, f'{name}_model.pth')
        torch.save(state, ckpt_path, pickle_protocol=4)
        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")

    def fit(self, train_data):
        """训练"""
        self.logger.info("="*60)
        self.logger.info("MM-RQVAE Training: Single paired DataLoader (text,image,index)")
        self.logger.info("="*60)

        for epoch_idx in range(self.epochs):
            training_start_time = time()
            total_loss, recon_loss, quant_loss, align_loss = self._train_epoch(
                train_data, epoch_idx
            )
            training_end_time = time()

            print(f'epoch {epoch_idx}, total: {total_loss:.4f}, recon: {recon_loss:.4f}, '
                  f'quant: {quant_loss:.4f}, align: {align_loss:.4f}')

            train_output = (
                set_color("epoch %d training", "green") + " [" +
                set_color("time", "blue") + ": %.2fs, " +
                set_color("total", "blue") + ": %.4f, " +
                set_color("recon", "blue") + ": %.4f, " +
                set_color("align", "blue") + ": %.4f]"
            ) % (epoch_idx, training_end_time - training_start_time, 
                 total_loss, recon_loss, align_loss)
            self.logger.info(train_output)

            # 更新 best_loss
            if total_loss < self.best_loss:
                self.best_loss = total_loss

            # 评估
            if (epoch_idx + 1) % self.eval_step == 0:
                collision_text, collision_image = self._valid_epoch(train_data)
                
                print(f'collision_text: {collision_text:.6f}, collision_image: {collision_image:.6f}')

                eval_output = (
                    set_color("epoch %d evaluating", "green") + " [" +
                    set_color("collision_text", "blue") + ": %.6f, " +
                    set_color("collision_image", "blue") + ": %.6f]"
                ) % (epoch_idx, collision_text, collision_image)
                self.logger.info(eval_output)

                # 保存最佳模型
                if collision_text < self.best_collision_text:
                    self.best_collision_text = collision_text
                    self._save_checkpoint(epoch_idx, 'best_text')

                if collision_image < self.best_collision_image:
                    self.best_collision_image = collision_image
                    self._save_checkpoint(epoch_idx, 'best_image')
                
                # LoRA 诊断（暂时关闭）
                # metrics = self.diagnostics.collect_metrics(epoch_idx)
                # self.diagnostics.log_metrics(epoch_idx, metrics)

        return self.best_loss, self.best_collision_text, self.best_collision_image
