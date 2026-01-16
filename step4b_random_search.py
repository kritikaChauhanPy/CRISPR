import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import random

print("="*60)
print("STEP 4B: RANDOM HYPERPARAMETER SEARCH")
print("Models: GRU, BiLSTM, CNN-BiLSTM, CNN-BiLSTM-Attention")
print("Configuration: 30 random configs per model, 20 epochs each")
print("="*60)

os.makedirs('results/random_search', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('models/random_search', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ========================================
# Load Data
# ========================================

print("\nLoading data...")
X_train = np.load('data/X_train.npy')
X_val = np.load('data/X_val.npy')
y_train = np.load('data/y_train.npy')
y_val = np.load('data/y_val.npy')

print(f"Train: {X_train.shape}, Positive: {np.sum(y_train==1)}")
print(f"Val:   {X_val.shape}, Positive: {np.sum(y_val==1)}")

pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"Class weight: {pos_weight:.2f}")

# ========================================
# Dataset
# ========================================

class CRISPRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CRISPRDataset(X_train, y_train)
val_dataset = CRISPRDataset(X_val, y_val)

# ========================================
# Hyperparameter Search Space
# ========================================

search_space = {
    'hidden_size': [64, 128, 256, 512, 768],
    'cnn_filters': [32, 64, 128, 256],
    'num_layers': [1, 2, 3, 4],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],
    'batch_size': [32, 64, 128, 256],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],
}

print("\n" + "="*60)
print("HYPERPARAMETER SEARCH SPACE")
print("="*60)
for param, values in search_space.items():
    print(f"{param:20} | {len(values)} options: {values}")

total_combinations = 1
for values in search_space.values():
    total_combinations *= len(values)
print(f"\nTotal possible combinations: {total_combinations:,}")
print(f"Testing: 30 random samples per model")

# ========================================
# Model Definitions
# ========================================

class ConfigurableGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(ConfigurableGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        fc1_out = self.relu(self.fc1(last_output))
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)
        
        return output

class ConfigurableBiLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(ConfigurableBiLSTM, self).__init__()
        
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        bilstm_out, (h_n, c_n) = self.bilstm(x)
        last_output = bilstm_out[:, -1, :]
        
        fc1_out = self.relu(self.fc1(last_output))
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)
        
        return output

class ConfigurableCNNBiLSTM(nn.Module):
    """Multi-scale CNN-BiLSTM (Main Contribution)"""
    def __init__(self, input_channels=4, cnn_filters=64, lstm_hidden=128, num_layers=2, dropout=0.3):
        super(ConfigurableCNNBiLSTM, self).__init__()
        
        # Multi-scale CNNs (kernels 3, 5, 7 for different motif lengths)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=7, padding=3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=cnn_filters * 3,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(x))
        conv3_out = self.relu(self.conv3(x))
        
        cnn_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        cnn_out = self.dropout(cnn_out)
        
        cnn_out = cnn_out.transpose(1, 2)
        
        bilstm_out, (h_n, c_n) = self.bilstm(cnn_out)
        last_output = bilstm_out[:, -1, :]
        
        fc1_out = self.relu(self.fc1(last_output))
        fc1_out = self.dropout(fc1_out)
        
        output = self.fc2(fc1_out)
        
        return output

class ConfigurableCNNBiLSTMAttention(nn.Module):
    """Multi-scale CNN-BiLSTM-Attention (Best Model with Built-in Interpretability)"""
    def __init__(self, input_channels=4, cnn_filters=64, lstm_hidden=128, num_layers=2, dropout=0.3):
        super(ConfigurableCNNBiLSTMAttention, self).__init__()
        
        # Multi-scale CNNs (kernels 3, 5, 7)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=7, padding=3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=cnn_filters * 3,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for interpretability
        self.attention = nn.Linear(lstm_hidden * 2, 1)
        
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(x))
        conv3_out = self.relu(self.conv3(x))
        
        cnn_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        cnn_out = self.dropout(cnn_out)
        
        cnn_out = cnn_out.transpose(1, 2)
        
        bilstm_out, (h_n, c_n) = self.bilstm(cnn_out)
        
        # Attention mechanism
        attention_weights = self.attention(bilstm_out)
        attention_weights = self.softmax(attention_weights)
        
        context = torch.sum(attention_weights * bilstm_out, dim=1)
        
        fc1_out = self.relu(self.fc1(context))
        fc1_out = self.dropout(fc1_out)
        
        output = self.fc2(fc1_out)
        
        return output

# ========================================
# Training Function
# ========================================

def train_with_config(model_class, config, model_name, config_id):
    """Train a model with specific hyperparameter configuration"""
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    if model_name in ['GRU', 'BiLSTM']:
        model = model_class(
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    else:  # CNN-BiLSTM models
        model = model_class(
            cnn_filters=config['cnn_filters'],
            lstm_hidden=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    pos_weight_tensor = torch.FloatTensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Training loop
    num_epochs = 20
    patience = 5
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        
        val_loss = total_loss / len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_loss, len(history['train_loss']), best_model_state

# ========================================
# Random Search for Each Model
# ========================================

models_to_search = {
    'GRU': ConfigurableGRU,
    'BiLSTM': ConfigurableBiLSTM,
    'CNN-BiLSTM': ConfigurableCNNBiLSTM,
    'CNN-BiLSTM-Attention': ConfigurableCNNBiLSTMAttention
}

num_random_samples = 30

all_results = []

for model_name, model_class in models_to_search.items():
    print("\n" + "="*60)
    print(f"RANDOM SEARCH: {model_name}")
    print("="*60)
    
    model_results = []
    
    for i in range(num_random_samples):
        config = {
            'hidden_size': random.choice(search_space['hidden_size']),
            'cnn_filters': random.choice(search_space['cnn_filters']),
            'num_layers': random.choice(search_space['num_layers']),
            'dropout': random.choice(search_space['dropout']),
            'learning_rate': random.choice(search_space['learning_rate']),
            'batch_size': random.choice(search_space['batch_size']),
            'weight_decay': random.choice(search_space['weight_decay'])
        }
        
        print(f"\nConfig {i+1}/{num_random_samples}:")
        print(f"  hidden={config['hidden_size']}, cnn={config['cnn_filters']}, "
              f"layers={config['num_layers']}, dropout={config['dropout']:.2f}, "
              f"lr={config['learning_rate']}, bs={config['batch_size']}, "
              f"wd={config['weight_decay']}")
        
        start_time = time.time()
        
        try:
            best_val_loss, epochs_trained, model_state = train_with_config(
                model_class, config, model_name, i
            )
            
            training_time = time.time() - start_time
            
            result = {
                'model': model_name,
                'config_id': i,
                'best_val_loss': best_val_loss,
                'epochs_trained': epochs_trained,
                'training_time': training_time,
                **config
            }
            
            model_results.append(result)
            all_results.append(result)
            
            # Save best model checkpoint
            torch.save({
                'model_state_dict': model_state,
                'config': config,
                'val_loss': best_val_loss
            }, f'models/random_search/{model_name}_config_{i}.pth')
            
            print(f"  Best Val Loss: {best_val_loss:.4f} | Epochs: {epochs_trained} | Time: {training_time/60:.1f}min")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    # Save model-specific results
    model_df = pd.DataFrame(model_results)
    model_df = model_df.sort_values('best_val_loss')
    model_df.to_csv(f'results/random_search/{model_name}_random_search.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"TOP 5 CONFIGS FOR {model_name}:")
    print(f"{'='*60}")
    for idx, row in model_df.head(5).iterrows():
        print(f"Rank {model_df.index.get_loc(idx)+1}: Config {int(row['config_id'])}")
        print(f"  Val Loss: {row['best_val_loss']:.4f}")
        print(f"  hidden={int(row['hidden_size'])}, cnn={int(row['cnn_filters'])}, "
              f"layers={int(row['num_layers'])}, dropout={row['dropout']:.2f}")
        print(f"  lr={row['learning_rate']}, batch={int(row['batch_size'])}, wd={row['weight_decay']}")
        print()

# ========================================
# Save All Results
# ========================================

all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv('results/random_search/all_random_search_results.csv', index=False)

# ========================================
# Visualization
# ========================================

print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, model_name in enumerate(models_to_search.keys()):
    model_data = all_results_df[all_results_df['model'] == model_name]
    
    ax = axes[idx]
    ax.scatter(range(len(model_data)), model_data['best_val_loss'], alpha=0.6, s=40, c='steelblue', edgecolors='black')
    ax.set_xlabel('Config ID', fontweight='bold', fontsize=11)
    ax.set_ylabel('Best Validation Loss', fontweight='bold', fontsize=11)
    ax.set_title(f'{model_name} - Random Search', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    best_loss = model_data['best_val_loss'].min()
    ax.axhline(y=best_loss, color='red', linestyle='--', linewidth=2, label=f'Best: {best_loss:.4f}')
    ax.legend(fontsize=10)
    ax.set_ylim([0, min(model_data['best_val_loss'].max() * 1.1, 2.0)])

plt.suptitle('Random Hyperparameter Search Results\n30 Configurations per Model', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/step4b_random_search_results.png', dpi=300, bbox_inches='tight')
print("Saved: figures/step4b_random_search_results.png")
plt.close()

# ========================================
# Summary
# ========================================

print("\n" + "="*60)
print("RANDOM SEARCH COMPLETE")
print("="*60)

print("\n" + "="*60)
print("BEST CONFIGURATION PER MODEL")
print("="*60)

for model_name in models_to_search.keys():
    model_data = all_results_df[all_results_df['model'] == model_name]
    best_row = model_data.loc[model_data['best_val_loss'].idxmin()]
    
    print(f"\n{model_name}:")
    print(f"  Best Val Loss: {best_row['best_val_loss']:.4f}")
    print(f"  Config ID: {int(best_row['config_id'])}")
    print(f"  Hidden Size: {int(best_row['hidden_size'])}")
    print(f"  CNN Filters: {int(best_row['cnn_filters'])}")
    print(f"  Num Layers: {int(best_row['num_layers'])}")
    print(f"  Dropout: {best_row['dropout']:.2f}")
    print(f"  Learning Rate: {best_row['learning_rate']}")
    print(f"  Batch Size: {int(best_row['batch_size'])}")
    print(f"  Weight Decay: {best_row['weight_decay']}")

print("\n" + "="*60)
print("FILES SAVED:")
print("="*60)
print("  - results/random_search/all_random_search_results.csv")
print("  - results/random_search/[MODEL]_random_search.csv (per model)")
print("  - models/random_search/[MODEL]_config_[ID].pth (checkpoints)")
print("  - figures/step4b_random_search_results.png")

print("\n" + "="*60)

print("="*60)