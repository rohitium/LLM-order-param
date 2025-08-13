#!/usr/bin/env python3
"""
Trains GPT models of different sizes (10M, 1M, 0.1M, 10K) based on command-line arguments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

# Model configurations for different sizes
MODEL_CONFIGS = {
    '10M': {
        'n_embd': 384,
        'n_head': 6,
        'n_layer': 6,
        'description': '10 Million parameters'
    },
    '1M': {
        'n_embd': 128,
        'n_head': 4,
        'n_layer': 4,
        'description': '1 Million parameters'
    },
    '0.1M': {
        'n_embd': 64,
        'n_head': 2,
        'n_layer': 2,
        'description': '0.1 Million parameters'
    },
    '10K': {
        'n_embd': 32,
        'n_head': 2,
        'n_layer': 1,
        'description': '10 Thousand parameters'
    }
}

# Training hyperparameters (same for all models)
TRAINING_CONFIG = {
    'batch_size': 64,
    'block_size': 256,
    'max_iters': 20000,
    'eval_interval': 500,
    'learning_rate': 3e-4,
    'eval_iters': 200,
    'dropout': 0.2
}

def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def build_tokenizer(data_path):
    """Build tokenizer from text data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    def encode(s): return [stoi[c] for c in s]
    def decode(l): return ''.join([itos[i] for i in l])
    return encode, decode, len(chars)

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_batch(train_data, val_data, split, batch_size, block_size, device):
    """Get a batch of data."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    """Estimate training and validation loss."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model_size, data_path='text/input.txt', output_dir='models', text_output_dir='text'):
    """Train a GPT model of the specified size."""
    
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_size]
    print(f"Training {model_size} model: {config['description']}")
    print(f"Hyperparameters: n_embd={config['n_embd']}, n_head={config['n_head']}, n_layer={config['n_layer']}")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(1337)
    
    # Load and prepare data
    print(f"Loading data from {data_path}...")
    encode, decode, vocab_size = build_tokenizer(data_path)
    
    data = torch.tensor(encode(open(data_path, 'r', encoding='utf-8').read()), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Data loaded: {len(data)} total tokens, {len(train_data)} training tokens, {len(val_data)} validation tokens")
    
    # Create model
    print("Creating model...")
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=TRAINING_CONFIG['block_size'],
        dropout=TRAINING_CONFIG['dropout']
    )
    
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    for iter in range(TRAINING_CONFIG['max_iters']):
        if iter % TRAINING_CONFIG['eval_interval'] == 0:
            losses = estimate_loss(
                model, train_data, val_data, 
                TRAINING_CONFIG['batch_size'], 
                TRAINING_CONFIG['block_size'], 
                TRAINING_CONFIG['eval_iters'], 
                device
            )
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Save model weights every other eval_interval
            if iter % (2 * TRAINING_CONFIG['eval_interval']) == 0:
                checkpoint_path = os.path.join(output_dir, f'gpt_{model_size}_step_{iter}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
        
        # Training step
        xb, yb = get_batch(train_data, val_data, 'train', TRAINING_CONFIG['batch_size'], TRAINING_CONFIG['block_size'], device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Save final model
    final_model_path = os.path.join(output_dir, f'gpt_{model_size}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Generate sample text
    print("Generating sample text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = model.generate(context, max_new_tokens=10000, block_size=TRAINING_CONFIG['block_size'])
    generated_text_str = decode(generated_text[0].tolist())
    
    text_output_path = os.path.join(text_output_dir, f'out_{model_size}.txt')
    with open(text_output_path, 'w') as f:
        f.write(generated_text_str)
    print(f"Generated text saved to {text_output_path}")
    
    print(f"Training complete for {model_size} model!")

def main():
    parser = argparse.ArgumentParser(description='Train GPT models of different sizes')
    parser.add_argument('model_size', choices=list(MODEL_CONFIGS.keys()), 
                       help='Size of the model to train')
    parser.add_argument('--data', default='text/input.txt', 
                       help='Path to input text file')
    parser.add_argument('--output-dir', default='models', 
                       help='Directory to save model checkpoints')
    parser.add_argument('--text-output-dir', default='text', 
                       help='Directory to save generated text')
    parser.add_argument('--max-iters', type=int, 
                       help='Override max training iterations')
    parser.add_argument('--eval-interval', type=int, 
                       help='Override evaluation interval')
    parser.add_argument('--batch-size', type=int, 
                       help='Override batch size')
    parser.add_argument('--learning-rate', type=float, 
                       help='Override learning rate')
    
    args = parser.parse_args()
    
    # Override training config if specified
    if args.max_iters:
        TRAINING_CONFIG['max_iters'] = args.max_iters
    if args.eval_interval:
        TRAINING_CONFIG['eval_interval'] = args.eval_interval
    if args.batch_size:
        TRAINING_CONFIG['batch_size'] = args.batch_size
    if args.learning_rate:
        TRAINING_CONFIG['learning_rate'] = args.learning_rate
    
    try:
        train_model(args.model_size, args.data, args.output_dir, args.text_output_dir)
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
