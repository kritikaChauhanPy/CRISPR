import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*60)
print("STEP 3: BUILDING ALL MODEL ARCHITECTURES")
print("="*60)

os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

X_train = np.load('data/X_train.npy')
print(f"\nData shape: {X_train.shape}")
print(f"Input: (batch_size, sequence_length=23, channels=4)")

# ========================================
# MODEL 1: Simple RNN
# ========================================

class SimpleRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(SimpleRNN, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        rnn_out, h_n = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        
        fc1_out = self.relu(self.fc1(last_output))
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)  # NO SIGMOID
        
        return output

# ========================================
# MODEL 2: Simple LSTM
# ========================================

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(SimpleLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        fc1_out = self.relu(self.fc1(last_output))
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)  # NO SIGMOID
        
        return output

# ========================================
# MODEL 3: Simple GRU
# ========================================

class SimpleGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(SimpleGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
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
        output = self.fc2(fc1_out)  # NO SIGMOID
        
        return output

# ========================================
# MODEL 4: Simple BiLSTM
# ========================================

class SimpleBiLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(SimpleBiLSTM, self).__init__()
        
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
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
        output = self.fc2(fc1_out)  # NO SIGMOID
        
        return output

# ========================================
# MODEL 5: CNN Only
# ========================================

class CNNOnly(nn.Module):
    def __init__(self, input_channels=4, cnn_filters=64, dropout=0.3):
        super(CNNOnly, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=7, padding=3)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(cnn_filters * 3 * 11, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        conv1_out = self.relu(self.conv1(x))
        conv1_out = self.pool(conv1_out)
        
        conv2_out = self.relu(self.conv2(x))
        conv2_out = self.pool(conv2_out)
        
        conv3_out = self.relu(self.conv3(x))
        conv3_out = self.pool(conv3_out)
        
        cnn_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        
        fc1_out = self.relu(self.fc1(cnn_out))
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = self.relu(self.fc2(fc1_out))
        fc2_out = self.dropout(fc2_out)
        
        output = self.fc3(fc2_out)  # NO SIGMOID
        
        return output

# ========================================
# MODEL 6: CNN-BiLSTM Hybrid
# ========================================

class CNNBiLSTM(nn.Module):
    def __init__(self, input_channels=4, cnn_filters=64, lstm_hidden=128, dropout=0.3):
        super(CNNBiLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=7, padding=3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=cnn_filters * 3,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
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
        
        output = self.fc2(fc1_out)  # NO SIGMOID
        
        return output

# ========================================
# MODEL 7: CNN-BiLSTM with Attention
# ========================================

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_channels=4, cnn_filters=64, lstm_hidden=128, dropout=0.3):
        super(CNNBiLSTMAttention, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=7, padding=3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=cnn_filters * 3,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
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
        
        attention_weights = self.attention(bilstm_out)
        attention_weights = self.softmax(attention_weights)
        
        context = torch.sum(attention_weights * bilstm_out, dim=1)
        
        fc1_out = self.relu(self.fc1(context))
        fc1_out = self.dropout(fc1_out)
        
        output = self.fc2(fc1_out)  # NO SIGMOID
        
        return output

# ========================================
# Create all model instances
# ========================================

print("\nCreating all models...")

models = {
    'RNN': SimpleRNN(),
    'LSTM': SimpleLSTM(),
    'GRU': SimpleGRU(),
    'BiLSTM': SimpleBiLSTM(),
    'CNN-only': CNNOnly(),
    'CNN-BiLSTM': CNNBiLSTM(),
    'CNN-BiLSTM-Attention': CNNBiLSTMAttention()
}

print("\nModel Summary:")
print("-" * 60)

for name, model in models.items():
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{name:25} | Parameters: {total_params:,}")

print("\nTesting forward pass for all models...")
test_input = torch.randn(16, 23, 4)

for name, model in models.items():
    try:
        output = model(test_input)
        print(f"{name:25} | Output shape: {output.shape} | Status: OK")
    except Exception as e:
        print(f"{name:25} | Error: {str(e)}")

# ========================================
# VISUALIZATION 1: Model Comparison Table
# ========================================

print("\nCreating model comparison table...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

model_info = []
for name, model in models.items():
    total_params = sum(p.numel() for p in model.parameters())
    
    if 'RNN' in name and 'CNN' not in name:
        model_type = 'Sequential'
    elif 'CNN' in name and 'LSTM' not in name and 'BiLSTM' not in name:
        model_type = 'Spatial'
    else:
        model_type = 'Hybrid'
    
    if 'Attention' in name:
        features = 'CNN + BiLSTM + Attention'
    elif 'CNN-BiLSTM' in name:
        features = 'CNN + BiLSTM'
    elif 'BiLSTM' in name:
        features = 'Bidirectional LSTM'
    elif 'LSTM' in name:
        features = 'LSTM'
    elif 'GRU' in name:
        features = 'GRU'
    elif 'RNN' in name:
        features = 'RNN'
    else:
        features = 'Multi-scale CNN'
    
    model_info.append([name, model_type, features, f'{total_params:,}'])

table = ax.table(cellText=model_info,
                 colLabels=['Model', 'Type', 'Components', 'Parameters'],
                 cellLoc='left',
                 loc='center',
                 colWidths=[0.25, 0.2, 0.35, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(4):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(model_info) + 1):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        
        if 'Hybrid' in model_info[i-1][1]:
            table[(i, 1)].set_facecolor('#3498db')
            table[(i, 1)].set_text_props(weight='bold', color='white')

plt.title('Model Architecture Comparison', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('figures/step3_model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: figures/step3_model_comparison.png")
plt.close()

# ========================================
# VISUALIZATION 2: Parameter Counts
# ========================================

print("Creating parameter counts visualization...")

fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(models.keys())
param_counts = [sum(p.numel() for p in model.parameters()) for model in models.values()]

colors = []
for name in model_names:
    if 'Attention' in name:
        colors.append('#c0392b')
    elif 'CNN-BiLSTM' in name:
        colors.append('#e74c3c')
    elif 'CNN' in name:
        colors.append('#f39c12')
    elif 'BiLSTM' in name:
        colors.append('#3498db')
    else:
        colors.append('#95a5a6')

bars = ax.bar(range(len(model_names)), param_counts, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Parameter Counts Across Models', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars, param_counts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/step3_parameter_counts.png', dpi=300, bbox_inches='tight')
print("Saved: figures/step3_parameter_counts.png")
plt.close()

# ========================================
# Save model architectures
# ========================================

print("\nSaving model architectures...")

for name, model in models.items():
    filename = name.replace('-', '_').replace('+', '_').lower()
    torch.save(model.state_dict(), f'models/{filename}_init.pth')

print("Model architectures saved in models/ directory")

print("\n" + "="*60)
print("STEP 3 COMPLETE")
print("="*60)
print("\nCreated 7 models (NO SIGMOID BUG - FIXED):")
for i, name in enumerate(models.keys(), 1):
    print(f"  {i}. {name}")

print("\nGenerated 2 visualizations:")
print("  1. step3_model_comparison.png")
print("  2. step3_parameter_counts.png")

print("\nKey fix: All models output raw logits (no sigmoid)")
print("This works correctly with BCEWithLogitsLoss")