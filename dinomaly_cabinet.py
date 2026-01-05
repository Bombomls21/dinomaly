"""
Training Dinomaly for Cabinet Anomaly Detection
"""
import torch
import torch.nn as nn
from dataset import get_data_transforms
from torchvision. datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader

from models.uad import ViTill
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dataset import MVTecDataset
import argparse
from utils import evaluation_batch, global_cosine_hm_percent, WarmCosineScheduler
from functools import partial
from optimizers import StableAdamW
import warnings
import logging
from ptflops import get_model_complexity_info  # ← ADD THIS

warnings.filterwarnings("ignore")

def get_logger(name, save_path=None, level='INFO'):
    logger = logging. getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path. join(save_path, 'log.txt'))
        fileHandler. setFormatter(log_format)
        logger.addHandler(fileHandler)
    
    return logger

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_flops(model, input_size=(3, 392, 392), device='cuda'):
    """Calculate FLOPs and params using ptflops"""
    model_copy = model.cpu()
    
    try:
        flops, params = get_model_complexity_info(
            model_copy, 
            input_size,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
        model. to(device)
        return flops, params
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs:  {e}")
        model.to(device)
        return "N/A", "N/A"

def print_model_stats(model, trainable, input_size=(3, 392, 392), device='cuda'):
    """Print comprehensive model statistics"""
    print_fn('\n' + '='*70)
    print_fn('MODEL STATISTICS')
    print_fn('='*70)
    
    # Parameters
    total_params, trainable_params = count_parameters(model)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in trainable. parameters())
    frozen_params = total_params - trainable_params
    
    print_fn(f'Total Parameters:        {total_params:,} ({total_params/1e6:.2f}M)')
    print_fn(f'  ├─ Encoder (frozen):  {encoder_params:,} ({encoder_params/1e6:.2f}M)')
    print_fn(f'  ├─ Decoder (train):   {decoder_params:,} ({decoder_params/1e6:.2f}M)')
    print_fn(f'  └─ Frozen:            {frozen_params:,} ({frozen_params/1e6:.2f}M)')
    print_fn(f'Trainable Parameters:   {trainable_params: ,} ({trainable_params/1e6:.2f}M) ({trainable_params/total_params*100:.1f}%)')
    
    # FLOPs
    print_fn(f'\nComputing FLOPs...')
    flops, params = get_model_flops(model, input_size, device)
    print_fn(f'FLOPs:                  {flops}')
    print_fn(f'Params (ptflops):       {params}')
    
    # Memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        dummy_input = torch.randn(1, *input_size).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch. cuda.max_memory_reserved() / 1024**3
        
        print_fn(f'\nGPU Memory (single forward):')
        print_fn(f'  ├─ Allocated:          {allocated:.3f} GB')
        print_fn(f'  └─ Reserved:           {reserved:.3f} GB')
        
        model. train()
    
    print_fn('='*70 + '\n')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn. deterministic = True
    torch. backends.cudnn.benchmark = False

def train_cabinet():
    setup_seed(1)
    
    # ========== CONFIGURATION ==========
    total_iters = 10000
    batch_size = 1
    image_size = 448
    crop_size = 392
    
    # Data transforms
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    
    # ========== LOAD DATA ==========
    print_fn(f'Loading dataset from: {args.data_path}')
    
    train_path = os.path.join(args.data_path, 'train')
    test_path = args.data_path
    
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, 
                            gt_transform=gt_transform, phase="test")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, 
        num_workers=4, drop_last=True
    )
    test_dataloader = torch. utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    print_fn(f'Train images: {len(train_data)}')
    print_fn(f'Test images: {len(test_data)}')
    
    # ========== MODEL CONFIGURATION ==========
    encoder_name = 'dinov2reg_vit_base_14'
    
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    
    # Load encoder
    encoder = vit_encoder.load(encoder_name)
    
    if 'small' in encoder_name:  
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:  
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    
    # ========== BUILD MODEL ==========
    print_fn(f'Building model with encoder: {encoder_name}')
    
    # Bottleneck
    bottleneck = nn.ModuleList([
        bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)
    ])
    
    # Decoder
    decoder = []
    for i in range(8):
        blk = VitBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2
        )
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)
    
    # Complete model
    model = ViTill(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder, 
        target_layers=target_layers, mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder, 
        fuse_layer_decoder=fuse_layer_decoder
    )
    model = model.to(device)
    
    trainable = nn.ModuleList([bottleneck, decoder])
    
    # Initialize weights
    for m in trainable.modules():
        if isinstance(m, nn. Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # ========== PRINT MODEL STATISTICS ========== ← ADD THIS
    print_model_stats(model, trainable, input_size=(3, crop_size, crop_size), device=device)
    
    # ========== OPTIMIZER ==========
    optimizer = StableAdamW(
        [{'params': trainable.parameters()}],
        lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, 
        amsgrad=True, eps=1e-10
    )
    
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=2e-3, final_value=2e-4, 
        total_iters=total_iters, warmup_iters=100
    )
    
    # ========== TRAINING LOOP ==========
    print_fn('='*50)
    print_fn('Starting training...')
    print_fn('='*50)
    
    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        loss_list = []
        
        for img, label in train_dataloader:
            img = img.to(device)
            
            # Forward
            en, de = model(img)
            
            # Loss with hard mining
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            
            loss_list.append(loss.item())
            lr_scheduler.step()
            
            # ========== EVALUATION ==========
            if (it + 1) % 2000 == 0 or (it + 1) == total_iters:
                model.eval()
                print_fn(f'\n{"="*50}')
                print_fn(f'Evaluation at iteration {it + 1}/{total_iters}')
                print_fn(f'{"="*50}')
                
                results = evaluation_batch(
                    model, test_dataloader, device, 
                    max_ratio=0.01, resize_mask=256
                )
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                
                print_fn(f'Image-level:  ')
                print_fn(f'  AUROC: {auroc_sp:.4f}')
                print_fn(f'  AP:     {ap_sp:.4f}')
                print_fn(f'  F1:    {f1_sp:.4f}')
                print_fn(f'Pixel-level: ')
                print_fn(f'  AUROC: {auroc_px:.4f}')
                print_fn(f'  AP:    {ap_px:.4f}')
                print_fn(f'  F1:    {f1_px:. 4f}')
                print_fn(f'  AUPRO: {aupro_px:.4f}')
                
                # Save checkpoint
                save_path = os.path.join(args.save_dir, args.save_name)
                os.makedirs(save_path, exist_ok=True)
                checkpoint_path = os.path.join(save_path, f'model_iter{it+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print_fn(f'\n✅ Checkpoint saved:  {checkpoint_path}')
                
                model.train()
            
            it += 1
            if it == total_iters: 
                break
        
        # Epoch summary
        avg_loss = np.mean(loss_list)
        print_fn(f'Epoch [{epoch+1}] Iter [{it}/{total_iters}] Loss:  {avg_loss:.4f}')
    
    # ========== FINAL SAVE ==========
    final_path = os.path.join(args.save_dir, args.save_name, 'model_final.pth')
    torch.save(model.state_dict(), final_path)
    print_fn(f'\n{"="*50}')
    print_fn(f'✅ Training completed!')
    print_fn(f'✅ Final model saved: {final_path}')
    print_fn(f'{"="*50}')

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Train Dinomaly on Cabinet Dataset')
    parser.add_argument('--data_path', type=str, default='../datasets/cabinet')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='cabinet_dinomaly')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print_fn(f'Using device:  {device}')
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print_fn(f'GPU: {gpu_name}')
        print_fn(f'GPU Memory: {gpu_memory:.1f} GB')
    
    # Start training
    train_cabinet()
    # python dinomaly_cabinet.py --data_path ../datasets/cabinet --save_dir ./results --save_name cabinet_experiment_v1 --device cuda:0
