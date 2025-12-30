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
from check_alignment_gain import check_alignment_from_paired_loader


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
        
        # 对齐检查参数（如果 args 中没有这些参数，使用默认值）
        self.check_alignment = getattr(args, 'check_alignment', True)
        self.alignment_check_step = getattr(args, 'alignment_check_step', 10)  # 每 10 个 epoch 检查一次
        
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

        for batch_idx, (batch_text, batch_image, _) in enumerate(pbar):
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
            # 建议每 50 或 100 个 step 打印一次，刚开始训练时可以设为 10
            if batch_idx % 50 == 0: 
                print(f"\n--- [Raw LoRA Check] Step {epoch_idx}-{batch_idx} ---")
                
                # 检查 Text 模态的使用 LoRA 的层
                if hasattr(self.model, 'text_rq'):
                    # 获取实际使用 LoRA 的层索引
                    lora_layer_indices = []
                    if hasattr(self.args, 'lora_layers') and self.args.lora_layers:
                        try:
                            lora_layer_indices = [int(x.strip()) for x in self.args.lora_layers.split(',')]
                        except:
                            lora_layer_indices = []
                    
                    # 如果没有指定，尝试找到第一个使用 LoRA 的层
                    if not lora_layer_indices:
                        for i, vq in enumerate(self.model.text_rq.vq_layers):
                            if hasattr(vq, 'use_lora') and vq.use_lora:
                                lora_layer_indices = [i]
                                break
                    
                    # 检查第一个使用 LoRA 的层
                    if lora_layer_indices:
                        layer_idx = lora_layer_indices[0]
                        if layer_idx < len(self.model.text_rq.vq_layers):
                            vq = self.model.text_rq.vq_layers[layer_idx]

                            # 检查是否使用LoRA且具有必要的属性（只在使用LoRA的层存在）
                            if (vq.use_lora and 
                                hasattr(vq, 'lora_A') and vq.lora_A is not None):

                                # 检查是否使用 Recursive LoRA
                                use_recursive = getattr(self.model.text_rq, 'use_recursive_lora', False)

                                with torch.no_grad():
                                    # 1. 【量级对比】
                                    # Base 码本的平均强度 (通常在 0.1 ~ 1.0 之间)
                                    base_norm = vq.embedding.weight.norm(dim=1).mean().item()

                                    # 获取当前的 B 矩阵（支持 Recursive LoRA）
                                    lora_B = vq.get_lora_B()
                                    if lora_B is not None:
                                        # LoRA 偏置的原始强度 (A @ B)
                                        lora_bias = torch.matmul(vq.lora_A, lora_B)
                                        lora_norm = lora_bias.norm(dim=1).mean().item()

                                        # 计算占比 (LoRA 到底是主力还是噪音？)
                                        ratio = (lora_norm / (base_norm + 1e-6)) * 100

                                        # 2. 【梯度心跳】
                                        grad_A = 0.0
                                        if vq.lora_A.grad is not None:
                                            grad_A = vq.lora_A.grad.norm().item()

                                        if use_recursive:
                                            # Recursive LoRA: 检查 B_init 和 evolution_network 的梯度
                                            grad_B_init = 0.0
                                            if hasattr(self.model.text_rq, 'B_init') and self.model.text_rq.B_init.grad is not None:
                                                grad_B_init = self.model.text_rq.B_init.grad.norm().item()

                                            grad_evol = 0.0
                                            if hasattr(self.model.text_rq, 'evolution_network'):
                                                for param in self.model.text_rq.evolution_network.parameters():
                                                    if param.grad is not None:
                                                        grad_evol += param.grad.norm().item()

                                            grad_B = grad_B_init + grad_evol

                                            # 3. 【打印报告】
                                            print(set_color(f"1. [强度] Base: {base_norm:.6f} | LoRA: {lora_norm:.8f}", "cyan"))
                                            print(set_color(f"   => LoRA 贡献占比: {ratio:.6f}%", "yellow"))
                                            print(set_color(f"2. [梯度] Grad_A: {grad_A:.8f} | Grad_B_init: {grad_B_init:.8f} | Grad_evol: {grad_evol:.8f}", "cyan"))
                                        else:
                                            # 标准 LoRA: 检查 lora_B 的梯度
                                            grad_B = 0.0
                                            if hasattr(vq, 'lora_B') and vq.lora_B is not None and vq.lora_B.grad is not None:
                                                grad_B = vq.lora_B.grad.norm().item()

                                            # 3. 【打印报告】
                                            print(set_color(f"1. [强度] Base: {base_norm:.6f} | LoRA: {lora_norm:.8f}", "cyan"))
                                            print(set_color(f"   => LoRA 贡献占比: {ratio:.6f}%", "yellow"))
                                            print(set_color(f"2. [梯度] Grad_A: {grad_A:.8f} | Grad_B: {grad_B:.8f}", "cyan"))

                                        # 4. 【自动判别】
                                        if ratio < 0.1:
                                            print(set_color("   [诊断] 🔴 蚂蚁撼树：LoRA 数值太小，被 Base 淹没了！(建议：大幅增大 A 的初始化)", "red"))
                                        elif ratio > 20.0:
                                            print(set_color("   [诊断] 🔴 喧宾夺主：LoRA 数值太大，可能在破坏特征！(建议：减小学习率)", "red"))
                                        elif grad_B == 0:
                                            print(set_color("   [诊断] 💀 梯度断联：B 没有收到梯度，检查代码逻辑。", "red"))
                                        else:
                                            print(set_color("   [诊断] 🟢 状态健康：数值在一个合理的辅助范围内。", "green"))
                                    else:
                                        print(set_color(f"   [警告] Layer {layer_idx} 的 B 矩阵为 None", "yellow"))
                            else:
                                print(set_color(f"   [警告] Layer {layer_idx} 没有 LoRA 参数", "yellow"))
                        else:
                            print(set_color(f"   [警告] Layer {layer_idx} 超出范围", "yellow"))
                    else:
                        print(set_color("   [警告] 没有找到使用 LoRA 的层", "yellow"))
            # ==========================================================
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
            # 打印 Gate 的值
            """
            if hasattr(self.model, 'text_rq'):
                # 获取文本模态每一层的 gate
                text_gates = [
                    round(vq.gate.item(), 4) 
                    for vq in self.model.text_rq.vq_layers 
                    if hasattr(vq, 'gate')
                ]
                # 获取图像模态每一层的 gate
                image_gates = [
                    round(vq.gate.item(), 4) 
                    for vq in self.model.image_rq.vq_layers 
                    if hasattr(vq, 'gate')
                ]
                self.logger.info(f"Text Gates: {text_gates}")
                self.logger.info(f"Image Gates: {image_gates}")
            """
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

                # 对齐增益检查（如果启用且在检查周期）
                if self.check_alignment and (epoch_idx + 1) % self.alignment_check_step == 0:
                    try:
                        self.logger.info(set_color("="*60, "yellow"))
                        self.logger.info(set_color(f"Checking alignment gain at epoch {epoch_idx}", "yellow"))
                        self.logger.info(set_color("="*60, "yellow"))
                        alignment_results = check_alignment_from_paired_loader(
                            self.model, train_data, self.device
                        )
                        # 记录结果到日志
                        self.logger.info(
                            f"Alignment Check Results: "
                            f"Base={alignment_results['sim_base']:.6f}, "
                            f"LoRA={alignment_results['sim_lora']:.6f}, "
                            f"Gain={alignment_results['gain']:.6f} "
                            f"({alignment_results['gain_percent']:.2f}%)"
                        )
                        self.logger.info(set_color("="*60, "yellow"))
                    except Exception as e:
                        self.logger.warning(f"Alignment check failed: {e}")

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
