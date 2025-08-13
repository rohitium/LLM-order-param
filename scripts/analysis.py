#!/usr/bin/env python3
"""
Analyzes GPT model checkpoints to compute order parameters, losses, and generate plots.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re
from pathlib import Path

from gpt import (
    GPTLanguageModel, 
    MODEL_CONFIGS, 
    build_tokenizer, 
    get_batch, 
    get_device
)

def compute_order_parameter_b(model, device):
    """Compute the order parameter b = ||W_lm_head||_F / n_embd."""
    with torch.no_grad():
        # Get the language model head weights
        lm_head_weights = model.lm_head.weight
        
        # Compute Frobenius norm
        frob_norm = torch.norm(lm_head_weights, p='fro')
        
        # Get embedding dimension
        n_embd = model.token_embedding_table.embedding_dim
        
        # Compute order parameter
        order_param = frob_norm / n_embd
        
        return order_param.item()

def compute_hidden_magnetization(model, data_ids, block_size, device, iters=50, batch_size=32):
    """Compute hidden state magnetization."""
    model.eval()
    magnetizations = []
    
    with torch.no_grad():
        for _ in range(iters):
            # Get random batch
            start_idx = torch.randint(0, len(data_ids) - block_size, (1,)).item()
            x = data_ids[start_idx:start_idx + block_size].unsqueeze(0).to(device)
            
            # Forward pass to get hidden states (before final layer norm and lm_head)
            tok_embd = model.token_embedding_table(x)
            pos_embd = model.position_embedding_table(torch.arange(block_size, device=device))
            hidden_states = tok_embd + pos_embd
            
            # Pass through transformer blocks
            for block in model.blocks:
                hidden_states = block(hidden_states)
            
            # Compute magnetization (norm of hidden states)
            magnetization = torch.norm(hidden_states, p='fro') / (block_size * hidden_states.shape[-1])
            magnetizations.append(magnetization.item())
    
    return np.mean(magnetizations)

def compute_val_loss(model, data_ids, block_size, device, iters=50, batch_size=32):
    """Compute validation loss."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for _ in range(iters):
            # Get random batch
            start_idx = torch.randint(0, len(data_ids) - block_size, (1,)).item()
            x = data_ids[start_idx:start_idx + block_size].unsqueeze(0).to(device)
            y = data_ids[start_idx + 1:start_idx + block_size + 1].unsqueeze(0).to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            losses.append(loss.item())
    
    return np.mean(losses)

def compute_train_loss(model, data_ids, block_size, device, iters=50, batch_size=32):
    """Compute training loss."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for _ in range(iters):
            # Get random batch
            start_idx = torch.randint(0, len(data_ids) - block_size, (1,)).item()
            x = data_ids[start_idx:start_idx + block_size].unsqueeze(0).to(device)
            y = data_ids[start_idx + 1:start_idx + block_size + 1].unsqueeze(0).to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            losses.append(loss.item())
    
    return np.mean(losses)

def analyze_model(model_size, checkpoint_pattern, device):
    """Analyze a single model's checkpoints."""
    print(f"Analyzing {model_size} model...")
    
    # Get model configuration
    config = MODEL_CONFIGS[model_size]
    
    # Find checkpoints
    checkpoint_files = glob.glob(checkpoint_pattern)
    checkpoint_files.sort()
    
    if not checkpoint_files:
        print(f"No checkpoints found for {model_size} model")
        return None
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Load text data
    encode, decode, vocab_size = build_tokenizer("text/input.txt")
    text = open("text/input.txt", "r").read()
    data_ids = torch.tensor(encode(text), dtype=torch.long)
    
    # Split data into training and validation sets (90% train, 10% val)
    split_idx = int(0.9 * len(data_ids))
    train_ids = data_ids[:split_idx]
    val_ids = data_ids[split_idx:]
    
    results = []
    
    for checkpoint_file in checkpoint_files:
        # Extract step number from filename
        match = re.search(r"step_(\d+)", checkpoint_file)
        if match:
            step = int(match.group(1))
        else:
            step = 0
        
        print(f"  Processing step {step}...")
        
        try:
            # Create model
            model = GPTLanguageModel(
                vocab_size=65,
                n_embd=config['n_embd'],
                n_head=config['n_head'],
                n_layer=config['n_layer'],
                block_size=256,
                dropout=0.2
            )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            model.to(device)
            
            # Compute metrics
            order_param_b = compute_order_parameter_b(model, device)
            hidden_mag = compute_hidden_magnetization(model, val_ids, 256, device)
            val_loss = compute_val_loss(model, val_ids, 256, device)
            train_loss = compute_train_loss(model, train_ids, 256, device)
            
            results.append({
                'model_size': model_size,
                'step': step,
                'order_parameter_b': order_param_b,
                'hidden_magnetization': hidden_mag,
                'val_loss': val_loss,
                'train_loss': train_loss
            })
            
            print(f"    Step {step}: b={order_param_b:.4f}, val_loss={val_loss:.4f}, train_loss={train_loss:.4f}")
            
        except Exception as e:
            print(f"    Error processing {checkpoint_file}: {e}")
            continue
    
    return results

def generate_order_parameter_plots(all_results):
    """Generate order parameter plots."""
    if not all_results:
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Order parameter b vs training steps
    for model_size, results in all_results.items():
        if results:
            steps = [r['step'] for r in results]
            order_params = [r['order_parameter_b'] for r in results]
            ax1.plot(steps, order_params, 'o-', label=f'{model_size}', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Order Parameter b')
    ax1.set_title('Order Parameter b vs Training Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot 2: Hidden magnetization vs training steps
    for model_size, results in all_results.items():
        if results:
            steps = [r['step'] for r in results]
            hidden_mags = [r['hidden_magnetization'] for r in results]
            ax2.plot(steps, hidden_mags, 'o-', label=f'{model_size}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Hidden Magnetization')
    ax2.set_title('Hidden Magnetization vs Training Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig("results/order_parameters_vs_steps.png", dpi=300, bbox_inches='tight', facecolor='#f8f9fa', edgecolor='none')
    plt.close()
    
    # Save CSV
    rows = []
    for model_size, results in all_results.items():
        for result in results:
            rows.append(result)
    
    df = pd.DataFrame(rows)
    df.to_csv("results/order_parameters_analysis.csv", index=False)
    print("Generated order parameter plots and CSV")

def generate_loss_plots(all_results):
    """Generate loss plots."""
    if not all_results:
        return
    
    # Create figure for 10M model training and validation loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: 10M model training and validation loss
    if '10M' in all_results and all_results['10M']:
        results_10M = all_results['10M']
        steps = [r['step'] for r in results_10M]
        val_losses = [r['val_loss'] for r in results_10M]
        train_losses = [r['train_loss'] for r in results_10M]
        
        ax1.plot(steps, val_losses, 'o-', label='Validation Loss', color='red', linewidth=2, markersize=6)
        ax1.plot(steps, train_losses, 's-', label='Training Loss', color='blue', linewidth=2, markersize=6)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('GPT-10M: Training and Validation Loss vs Steps')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
    
    # Plot 2: All models validation loss
    for model_size, results in all_results.items():
        if results:
            steps = [r['step'] for r in results]
            val_losses = [r['val_loss'] for r in results]
            ax2.plot(steps, val_losses, 'o-', label=f'{model_size}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('All Models: Validation Loss vs Training Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig("results/val-train-loss-gpt-10M.png", dpi=300, bbox_inches='tight', facecolor='#f8f9fa', edgecolor='none')
    plt.savefig("results/all_models_validation_loss.png", dpi=300, bbox_inches='tight', facecolor='#f8f9fa', edgecolor='none')
    plt.close()
    
    print("Generated loss plots")

def generate_derivative_plots():
    """Generate derivative plots by reading existing CSV data."""
    # Read the existing derivative data
    try:
        df = pd.read_csv("results/all_models_loss_derivative.csv")
        print("Loaded existing derivative data from CSV")
    except FileNotFoundError:
        print("Derivative CSV not found, skipping derivative plots")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot first derivative for each model
    for model_size in ['10K', '0.1M', '1M', '10M']:
        model_data = df[df['model_size'] == model_size]
        if not model_data.empty:
            # Skip first step for each model
            model_data = model_data.iloc[1:]
            ax.plot(model_data['order_parameter_b'], model_data['dL_db'], 
                   'o-', label=f'{model_size}', linewidth=2, markersize=6)
    
    # Add vertical lines at specific points
    ax.axvline(x=0.14, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=0.334, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Order Parameter b')
    ax.set_ylabel('dL/db')
    ax.set_title('First Derivative of Loss w.r.t Order Parameter b')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig("results/all_models_first_derivative_with_lines.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated derivative plots")

def generate_train_loss_vs_order_parameter_plots():
    """Generate train loss vs order parameter plots by reading existing CSV data."""
    # Read the existing data
    try:
        df = pd.read_csv("results/all_models_train_loss_vs_order_parameter.csv")
        print("Loaded existing train loss vs order parameter data from CSV")
    except FileNotFoundError:
        print("Train loss vs order parameter CSV not found, skipping plots")
        return
    
    # Create the main plot (no title)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot for each model with specific legend order
    legend_order = ['10K', '0.1M', '1M', '10M']
    colors = ['red', 'blue', 'green', 'purple']
    
    for i, model_size in enumerate(legend_order):
        model_data = df[df['model_size'] == model_size]
        if not model_data.empty:
            ax.plot(model_data['order_parameter_b'], model_data['train_loss'], 
                   'o-', label=f'{model_size}', color=colors[i], linewidth=2, markersize=6)
    
    ax.set_xlabel('Order Parameter b')
    ax.set_ylabel('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig("results/all_models_train_loss_vs_order_parameter_no_title.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create subplots version
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, model_size in enumerate(legend_order):
        model_data = df[df['model_size'] == model_size]
        if not model_data.empty:
            axes[i].plot(model_data['order_parameter_b'], model_data['train_loss'], 
                        'o-', color=colors[i], linewidth=2, markersize=6)
            axes[i].set_xlabel('Order Parameter b')
            axes[i].set_ylabel('Training Loss')
            axes[i].set_title(f'{model_size} Model')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig("results/all_models_train_loss_vs_order_parameter_subplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated train loss vs order parameter plots")

def generate_model_comparison(all_results):
    """Generate model comparison summary."""
    summary_rows = []
    
    for model_size, results in all_results.items():
        if results:
            # Get final results (last checkpoint)
            final_result = max(results, key=lambda x: x['step'])
            
            summary_rows.append({
                'model_size': model_size,
                'final_step': final_result['step'],
                'final_order_parameter_b': final_result['order_parameter_b'],
                'final_hidden_magnetization': final_result['hidden_magnetization'],
                'final_val_loss': final_result['val_loss'],
                'final_train_loss': final_result['train_loss'],
                'total_checkpoints': len(results)
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv("results/model_summary.csv", index=False)
        print("Generated model comparison summary CSV")

def analyze_all_models():
    """Analyze all available models."""
    print("Starting analysis of all models...")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Model patterns
    model_patterns = {
        '10M': 'models/gpt_10M_step_*.pth',
        '1M': 'models/gpt_1M_step_*.pth',
        '0.1M': 'models/gpt_0.1M_step_*.pth',
        '10K': 'models/gpt_10K_step_*.pth'
    }
    
    all_results = {}
    
    # Analyze each model
    for model_size, pattern in model_patterns.items():
        results = analyze_model(model_size, pattern, device)
        if results:
            all_results[model_size] = results
    
    # Generate plots and summaries
    if all_results:
        generate_order_parameter_plots(all_results)
        generate_loss_plots(all_results)
        generate_derivative_plots()
        generate_train_loss_vs_order_parameter_plots()
        generate_model_comparison(all_results)
        print("Analysis complete!")
    else:
        print("No results to analyze")

if __name__ == "__main__":
    analyze_all_models()
