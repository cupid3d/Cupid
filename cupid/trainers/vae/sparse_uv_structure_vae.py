from typing import *
import copy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from .sparse_structure_vae import *
from ...utils.pose_utils import *


class SparseUVStructureVaeTrainer(BasicTrainer):
    """
    Trainer for Sparse UV Structure VAE.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        classification_loss_type (str): Loss type. 'bce' for binary cross entropy, 'l1' for L1 loss, 'dice' for Dice loss.
        regression_loss_type (str): Loss type. 'l1' for L1 loss. 'mse' for MSE loss. 'huber' for Huber loss.
        uv_loss_mask (str): Mask type to apply UV regression loss. Options are ['ss', 'ssuv', 'full'].
        input_ssuv (bool): Whether to use the sparse UV structure (partial voxels) as input. Default is True.
        input_uv_volume (bool): Whether to use the uv volume as input. Default is True.
        input_depth (bool): Whether to use the depth volume as input. Default is False.
        predict_ssuv (bool): Whether to predict the sparse UV structure. Default is True.
        predict_confidence (bool): Whether to predict the confidence of the UV volume. Default is False.
        lambda_kl (float): KL divergence loss weight.
        lambda_uv_volume (float): UV volume regression loss weight.
    """
    
    def __init__(
        self,
        *args,
        classification_loss_type='bce',
        regression_loss_type='l1',
        uv_loss_mask='ssuv',
        input_ssuv=True,
        input_uv_volume=True,
        input_depth=False,
        predict_ssuv=True,
        predict_confidence=False,
        lambda_kl=1e-6,
        lambda_uv_volume=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.classification_loss_type = classification_loss_type
        self.regression_loss_type = regression_loss_type
        self.uv_loss_mask = uv_loss_mask
        self.input_ssuv = input_ssuv
        self.input_uv_volume = input_uv_volume
        self.input_depth = input_depth
        self.predict_ssuv = predict_ssuv
        self.predict_confidence = predict_confidence
        self.lambda_kl = lambda_kl
        self.lambda_uv_volume = lambda_uv_volume
        self._xyz_voxels = None

    def add_binary_classification_loss(self, terms, name: str, logits: torch.Tensor, target: torch.Tensor):
        target = target.float()
        if self.classification_loss_type == 'bce':
            terms[f"{name}_bce"] = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
            terms["loss"] = terms["loss"] + terms[f"{name}_bce"]
        elif self.classification_loss_type == 'l1':
            terms[f"{name}_l1"] = F.l1_loss(F.sigmoid(logits), target, reduction='mean')
            terms["loss"] = terms["loss"] + terms[f"{name}_l1"]
        elif self.classification_loss_type == 'dice':
            logits = F.sigmoid(logits)
            terms[f"{name}_dice"] = 1 - (2 * (logits * target).sum() + 1) / (logits.sum() + target.sum() + 1)
            terms["loss"] = terms["loss"] + terms[f"{name}_dice"]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

    def add_regression_loss(self, terms, name: str, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, lambda_weight: float = 1.0):
        # Apply weight mask - only compute loss where weight is non-zero
        if weight.sum() > 0:
            if self.regression_loss_type == 'l1':
                terms[f"{name}_l1"] = (F.l1_loss(pred, target, reduction='none') * weight).sum() / weight.sum()
                terms["loss"] = terms["loss"] + lambda_weight * terms[f"{name}_l1"]
            elif self.regression_loss_type == 'mse':
                terms[f"{name}_mse"] = (F.mse_loss(pred, target, reduction='none') * weight).sum() / weight.sum()
                terms["loss"] = terms["loss"] + lambda_weight * terms[f"{name}_mse"]
            elif self.regression_loss_type == 'huber':
                terms[f"{name}_huber"] = (F.huber_loss(pred, target, reduction='none') * weight).sum() / weight.sum()
                terms["loss"] = terms["loss"] + lambda_weight * terms[f"{name}_huber"]
            else:
                raise ValueError(f'Invalid loss type {self.regression_loss_type}')
        else:
            # If no valid weight positions, set loss to 0
            terms[f"{name}_{self.regression_loss_type}"] = torch.tensor(0.0, device=pred.device)
            terms["loss"] = terms["loss"] + lambda_weight * terms[f"{name}_{self.regression_loss_type}"]

    def training_losses(
        self,
        ss: torch.Tensor,
        ssuv: torch.Tensor,
        uv_volume: torch.Tensor,
        **extra_data
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            ss: The [N x 1 x H x W x D] tensor of binary sparse structure.
            ssuv: The [N x 1 x H x W x D] tensor of binary sparse structure with UV.
            uv_volume: The [N x 2 x H x W x D] tensor of UV coordinates.
            **extra_data: Other optional data, may contain the following keys:
                - 'intrinsics': The [N x 3 x 3] tensor of intrinsics.
                - 'extrinsics': The [N x 4 x 4] tensor of extrinsics.
                - 'depth_volume': The [N x 1 x H x W x D] tensor of depth volume.
                - 'cond': The [N x 3 x ImageHeight x ImageWidth] tensor of image condition.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """

        inputs = []
        if self.input_ssuv:
            inputs.append(ssuv.float())
        if self.input_uv_volume:
            inputs.append(uv_volume)
        if self.input_depth:
            inputs.append(extra_data['depth_volume'])

        input_volume = torch.cat(inputs, dim=1)
        z, mean, logvar = self.training_models['encoder'](input_volume, sample_posterior=True, return_raw=True)
        logits = self.training_models['decoder'](z)

        if self.predict_ssuv:
            ssuv_logits, logits = logits[:, :1, :, :, :], logits[:, 1:, :, :, :]
        uv_volume_pred = torch.sigmoid(logits[:, :2, :, :, :])

        terms = edict(loss = 0.0)
        if self.predict_ssuv:
            self.add_binary_classification_loss(terms, "ssuv", ssuv_logits, ssuv)
        uv_loss_weight = {
            'ss': ss,
            'ssuv': ssuv,
            'full': torch.ones_like(ssuv),
        }[self.uv_loss_mask]
        self.add_regression_loss(terms, "uv_volume", uv_volume_pred, uv_volume, uv_loss_weight, self.lambda_uv_volume)

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
    
        return terms, {}
    
    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=1, verbose=False):
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        ssuv_gts = []
        uv_volume_gts = []
        ssuv_recons = []
        uv_volume_preds = []

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            
            # Prepare input volume: concatenate ss, ssuv, uv_volume, uv_weight
            inputs = []
            if self.input_ssuv:
                inputs.append(args['ssuv'].float())
            if self.input_uv_volume:
                inputs.append(args['uv_volume'])
            if self.input_depth:
                inputs.append(args['depth_volume'])
            
            # Run inference
            input_volume = torch.cat(inputs, dim=1)
            z = self.models['encoder'](input_volume, sample_posterior=False)
            logits = self.models['decoder'](z)
            
            # Split outputs
            if self.predict_ssuv:
                ssuv_logits, logits = logits[:, :1, :, :, :], logits[:, 1:, :, :, :]
            uv_volume_pred = torch.sigmoid(logits[:, :2, :, :, :])
            
            # Store ground truth and reconstructions
            ssuv_gts.append(args['ssuv'])
            if self.predict_ssuv:
                ssuv_recons.append((ssuv_logits > 0).int())
            uv_volume_gts.append(args['uv_volume'])
            uv_volume_preds.append(uv_volume_pred)
        
        ssuv_gts = torch.cat(ssuv_gts, dim=0)
        if self.predict_ssuv:
            ssuv_recons = torch.cat(ssuv_recons, dim=0)
        uv_volume_gts = torch.cat(uv_volume_gts, dim=0)
        uv_volume_preds = torch.cat(uv_volume_preds, dim=0)
        uv_volume_diffs = (uv_volume_gts - uv_volume_preds).norm(dim=1, keepdim=True)
        uv_volume_diffs = uv_volume_diffs / uv_volume_diffs.max().clamp_min_(1e-3)

        sample_dict = {}
        if self.predict_ssuv:
            sample_dict['ssuv_gt'] = {'value': ssuv_gts, 'type': 'sample'}
            sample_dict['ssuv_recon'] = {'value': ssuv_recons, 'type': 'sample'}
        sample_dict['uv_volume_gt'] = {'value': {'uv_volume': uv_volume_gts, 'ssuv': ssuv_gts}, 'type': 'sample'}
        sample_dict['uv_volume_pred'] = {'value': {'uv_volume': uv_volume_preds, 'ssuv': ssuv_gts}, 'type': 'sample'}
        sample_dict['uv_volume_diff'] = {'value': {'uv_volume': uv_volume_diffs, 'ssuv': ssuv_gts}, 'type': 'sample'}

        return sample_dict
