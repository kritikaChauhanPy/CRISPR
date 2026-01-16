"""
Final Test Evaluation - Option B (Improved)
Trains on 90% data (train+val combined) with internal monitoring for early stopping
Incorporates fixes: lower LR, weighted loss, proper early stopping
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ConfigurableGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = self.relu(self.fc1(gru_out[:, -1, :]))
        return self.fc2(self.dropout(x))

class ConfigurableBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)
        x = self.relu(self.fc1(bilstm_out[:, -1, :]))
        return self.fc2(self.dropout(x))

class CNNBiLSTM4ch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_filters, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout)
        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def train_final_model(model, X_combined, y_combined, config, model_name):
    """
    Train model on 90% data with internal 10% monitoring for early stopping
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nSplitting 90% data into 90% train + 10% internal monitor...")
    n_combined = len(X_combined)
    indices = np.random.permutation(n_combined)
    
    split_idx = int(0.9 * n_combined)
    train_indices = indices[:split_idx]
    monitor_indices = indices[split_idx:]
    
    X_train_internal = X_combined[train_indices]
    y_train_internal = y_combined[train_indices]
    X_monitor = X_combined[monitor_indices]
    y_monitor = y_combined[monitor_indices]
    
    print(f"  Internal training: {len(X_train_internal)} samples")
    print(f"  Internal monitor: {len(X_monitor)} samples")
    
    pos_count = sum(y_train_internal)
    neg_count = len(y_train_internal) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    
    print(f"  Class imbalance: {neg_count/pos_count:.2f}:1 (neg:pos)")
    print(f"  Positive weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')
    
    best_monitor_auprc = 0
    patience = 10
    patience_counter = 0
    max_epochs = 100
    best_model_state = None
    
    print(f"\nTraining with early stopping (patience={patience})...")
    
    for epoch in range(max_epochs):
        model.train()
        indices_epoch = np.random.permutation(len(X_train_internal))
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train_internal), config['batch_size']):
            batch_indices = indices_epoch[i:i+config['batch_size']]
            batch_X = torch.FloatTensor(X_train_internal[batch_indices]).to(device)
            batch_y = torch.FloatTensor(y_train_internal[batch_indices]).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            if i % (config['batch_size'] * 10) == 0:
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / n_batches
        
        model.eval()
        with torch.no_grad():
            monitor_X = torch.FloatTensor(X_monitor).to(device)
            with autocast('cuda'):
                monitor_logits = model(monitor_X).cpu().numpy().flatten()
            monitor_probs = expit(monitor_logits)
            monitor_auprc = average_precision_score(y_monitor, monitor_probs)
        
        if monitor_auprc > best_monitor_auprc:
            best_monitor_auprc = monitor_auprc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Monitor AUPRC={monitor_auprc:.4f} ✓ (best)")
        else:
            patience_counter += 1
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Monitor AUPRC={monitor_auprc:.4f} (patience {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch+1}")
                print(f"  Best monitor AUPRC: {best_monitor_auprc:.4f}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{model_name.lower().replace("-", "_")}.pt')
    
    return model

def evaluate_model(model, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        test_X = torch.FloatTensor(X_test).to(device)
        from torch.amp import autocast
        with autocast('cuda'):
            test_logits = model(test_X).cpu().numpy().flatten()
    
    test_probs = expit(test_logits)
    
    auprc = average_precision_score(y_test, test_probs)
    auroc = roc_auc_score(y_test, test_probs)
    
    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    
    return {
        'auprc': auprc,
        'auroc': auroc,
        'precision_curve': precision,
        'recall_curve': recall,
        'fpr': fpr,
        'tpr': tpr,
        'predictions': test_probs
    }

print("="*80)
print("FINAL TEST EVALUATION - OPTION B (IMPROVED)")
print("="*80)
print("\nStrategy:")
print("  1. Combine train+val = 137,909 samples (90% of all data)")
print("  2. Split into 90% internal train + 10% internal monitor")
print("  3. Lower learning rates (0.5x original)")
print("  4. Weighted loss for class imbalance")
print("  5. Early stopping with patience=10")
print("  6. Test on held-out 15,324 samples (10%)")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\nLoading data...")
X_train = np.load('data/X_train.npy')
X_val = np.load('data/X_val.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_val = np.load('data/y_val.npy')
y_test = np.load('data/y_test.npy')

X_combined = np.concatenate([X_train, X_val], axis=0)
y_combined = np.concatenate([y_train, y_val], axis=0)

print(f"\nData splits:")
print(f"  Combined (train+val): {X_combined.shape[0]} samples (90% of total)")
print(f"  Test (held-out): {X_test.shape[0]} samples (10% of total)")

best_configs = {
    'GRU-4ch': {
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.25,
        'learning_rate': 0.0005,
        'batch_size': 64,
        'original_lr': 0.001
    },
    'BiLSTM-4ch': {
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.2,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'original_lr': 0.001
    },
    'CNN-BiLSTM-4ch': {
        'hidden_size': 192,
        'num_layers': 1,
        'dropout': 0.18,
        'learning_rate': 0.00025,
        'batch_size': 64,
        'num_filters': 64,
        'kernel_size': 3,
        'original_lr': 0.0005
    }
}

results = {}

for model_name, config in best_configs.items():
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    for key, value in config.items():
        if key != 'original_lr':
            print(f"  {key}: {value}")
    print(f"  Learning rate: {config['learning_rate']} (reduced from {config['original_lr']})")
    
    if model_name == 'GRU-4ch':
        model = ConfigurableGRU(4, config['hidden_size'], config['num_layers'], config['dropout'])
    elif model_name == 'BiLSTM-4ch':
        model = ConfigurableBiLSTM(4, config['hidden_size'], config['num_layers'], config['dropout'])
    else:
        model = CNNBiLSTM4ch(4, config['hidden_size'], config['num_layers'], 
                            config['dropout'], config['num_filters'], config['kernel_size'])
    
    model = train_final_model(model, X_combined, y_combined, config, model_name)
    
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"\nFinal Test Results:")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    
    results[model_name] = metrics

os.makedirs('results/test_evaluation', exist_ok=True)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test_AUPRC': [results[m]['auprc'] for m in results.keys()],
    'Test_AUROC': [results[m]['auroc'] for m in results.keys()]
})
results_df.to_csv('results/test_evaluation/final_test_results_optionB.csv', index=False)

print("\n" + "="*80)
print("GENERATING FIGURES")
print("="*80)

os.makedirs('figures', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
colors = ['#3498db', '#e74c3c', '#2ecc71']
for idx, (model_name, metrics) in enumerate(results.items()):
    ax.plot(metrics['recall_curve'], metrics['precision_curve'],
           label=f"{model_name.replace('-4ch', '')} (AUPRC={metrics['auprc']:.3f})",
           color=colors[idx], linewidth=2.5, alpha=0.85)

baseline = np.sum(y_test) / len(y_test)
ax.axhline(baseline, color='gray', linestyle='--', linewidth=2,
          label=f'Random (AUPRC={baseline:.3f})', alpha=0.7)

ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
ax.set_title('Precision-Recall Curves - Test Set', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

ax = axes[1]
models = list(results.keys())
auprc_values = [results[m]['auprc'] for m in models]
bars = ax.bar(range(len(models)), auprc_values, color=colors, alpha=0.8, width=0.6)

ax.set_ylabel('Test AUPRC', fontsize=13, fontweight='bold')
ax.set_title('Model Comparison - Test Performance', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(models)))
ax.set_xticklabels([m.replace('-4ch', '') for m in models], fontsize=11)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

for bar, value in zip(bars, auprc_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax = axes[2]
benchmarks = {
    'Rahman\nLSTM\n(2024)': 0.7208,
    'GRU\n(Ours)': results['GRU-4ch']['auprc'],
    'BiLSTM\n(Ours)': results['BiLSTM-4ch']['auprc'],
    'CNN-BiLSTM\n(Ours)': results['CNN-BiLSTM-4ch']['auprc']
}

benchmark_colors = ['gray'] + colors
bars = ax.bar(benchmarks.keys(), benchmarks.values(), 
             color=benchmark_colors, alpha=0.8)

ax.set_ylabel('Test AUPRC', fontsize=13, fontweight='bold')
ax.set_title('Comparison with State-of-the-Art', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)
ax.axhline(0.7208, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Rahman threshold')

for bar, (name, value) in zip(bars, benchmarks.items()):
    height = bar.get_height()
    color = 'green' if value > 0.7208 else 'red'
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('figures/final_test_evaluation_optionB.png', dpi=300, bbox_inches='tight')
print("Saved: figures/final_test_evaluation_optionB.png")
plt.close()

print("\n" + "="*80)
print("FINAL TEST EVALUATION COMPLETE")
print("="*80)

print("\nTest Results (trained on 90% data with internal monitoring):")
for model_name in results.keys():
    print(f"{model_name:20s}: AUPRC {results[model_name]['auprc']:.4f} | AUROC {results[model_name]['auroc']:.4f}")

print("\n" + "-"*80)
print("Comparison with Rahman et al. (2024):")
print(f"{'Model':<25s} {'Test AUPRC':<12s} {'Status':<15s} {'Difference'}")
print("-"*80)

rahman_auprc = 0.7208
print(f"{'Rahman LSTM':<25s} {rahman_auprc:<12.4f} {'Baseline':<15s} {'-'}")

for model_name in results.keys():
    auprc = results[model_name]['auprc']
    diff = auprc - rahman_auprc
    diff_pct = (diff / rahman_auprc) * 100
    
    if auprc > rahman_auprc:
        status = "✓ BEATS"
        status_str = f"{status} (+{diff_pct:.1f}%)"
    else:
        status = "Below"
        status_str = f"{status} ({diff_pct:.1f}%)"
    
    print(f"{model_name:<25s} {auprc:<12.4f} {status_str:<15s} {diff:+.4f}")

print("-"*80)

best_model = max(results.keys(), key=lambda m: results[m]['auprc'])
best_auprc = results[best_model]['auprc']

if best_auprc > 0.7208:
    print(f"\n✓ SUCCESS: {best_model} beats state-of-the-art!")
    print(f"  Your result: {best_auprc:.4f}")
    print(f"  Rahman 2024: {rahman_auprc:.4f}")
    print(f"  Improvement: {((best_auprc - rahman_auprc) / rahman_auprc * 100):.1f}%")
else:
    print(f"\nBest model: {best_model} with {best_auprc:.4f} AUPRC")
    print(f"Below Rahman by {((rahman_auprc - best_auprc) / rahman_auprc * 100):.1f}%")
    print("Consider: Report as solid contribution with honest analysis")

print("\nFiles saved:")
print("  results/test_evaluation/final_test_results_optionB.csv")
print("  figures/final_test_evaluation_optionB.png")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("Review test results above")
print("="*80)