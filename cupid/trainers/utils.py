import math
import torch
import torch.nn as nn

# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    """
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )
    

def zero_grad(model_params):
    for param in model_params:
       if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()


def handle_partial_mismatch(ckpt_tensor, model_tensor):
    """
    Handle partial mismatch by keeping the common part from checkpoint and 
    the excess part from the current model state dict.
    
    Args:
        ckpt_tensor: tensor from checkpoint
        model_tensor: tensor from current model state dict
        
    Returns:
        tensor: combined tensor with common part from checkpoint and excess part from model
    """
    if ckpt_tensor.ndim != model_tensor.ndim:
        raise ValueError(f'Tensor dimensions must match for partial loading. Got {ckpt_tensor.ndim} vs {model_tensor.ndim}')
    
    # Create a copy of the model tensor
    result = model_tensor.clone()
    
    # Create slicing indices for the common part
    slices = []
    for i in range(ckpt_tensor.ndim):
        if ckpt_tensor.shape[i] <= model_tensor.shape[i]:
            slices.append(slice(0, ckpt_tensor.shape[i]))
        else:
            raise ValueError(f'Checkpoint tensor dimension {i} ({ckpt_tensor.shape[i]}) is larger than model tensor dimension {i} ({model_tensor.shape[i]})')
    
    # Load the common part from checkpoint
    result[tuple(slices)] = ckpt_tensor
    
    return result


def handle_zeropad_mismatch(ckpt_tensor, model_tensor):
    """
    Handle zeropad mismatch by keeping the common part from checkpoint and 
    filling the excess part with zeros.
    
    Args:
        ckpt_tensor: tensor from checkpoint
        model_tensor: tensor from current model state dict
        
    Returns:
        tensor: combined tensor with common part from checkpoint and excess part filled with zeros
    """
    if ckpt_tensor.ndim != model_tensor.ndim:
        raise ValueError(f'Tensor dimensions must match for zeropad loading. Got {ckpt_tensor.ndim} vs {model_tensor.ndim}')
    
    # Create a zero tensor with the model's shape
    result = torch.zeros_like(model_tensor)
    
    # Create slicing indices for the common part
    slices = []
    for i in range(ckpt_tensor.ndim):
        if ckpt_tensor.shape[i] <= model_tensor.shape[i]:
            slices.append(slice(0, ckpt_tensor.shape[i]))
        else:
            raise ValueError(f'Checkpoint tensor dimension {i} ({ckpt_tensor.shape[i]}) is larger than model tensor dimension {i} ({model_tensor.shape[i]})')
    
    # Load the common part from checkpoint
    result[tuple(slices)] = ckpt_tensor
    
    return result 
            

# LR Schedulers
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

class LinearWarmupLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0


class CosineWarmupScheduler(LRScheduler):
    """
    Cosine warmup scheduler with cosine annealing.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        initial_lr: Initial learning rate for warmup (default: 1e-10)
        end_lr: End learning rate for cosine annealing (default: 0)
        last_epoch: Last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_steps, max_steps, initial_lr=1e-10, end_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            # Linear warmup
            return [
                self.initial_lr + (base_lr - self.initial_lr) * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            cos_iter = self._step_count - self.warmup_steps
            cos_max_iter = self.max_steps - self.warmup_steps
            cos_theta = cos_iter / cos_max_iter * math.pi
            cos_lr = [
                self.end_lr + (base_lr - self.end_lr) * (1 + math.cos(cos_theta)) / 2
                for base_lr in self.base_lrs
            ]
            return cos_lr        