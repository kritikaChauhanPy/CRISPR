import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, 
                            roc_curve, confusion_matrix, classification_report, f1_score, 
                            accuracy_score, recall_score, precision_score)
from scipy.special import expit
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

def train_model(model, config, X_train, y_train, X_val, y_val):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                 weight_decay=config.get('weight_decay', 0))
    
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    max_epochs = 20
    best_model_state = None
    
    for epoch in range(max_epochs):
        model.train()
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), config['batch_size']):
            batch_indices = indices[i:i+config['batch_size']]
            batch_X = torch.FloatTensor(X_train[batch_indices]).to(device)
            batch_y = torch.FloatTensor(y_train[batch_indices]).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_X = torch.FloatTensor(X_val).to(device)
            val_y = torch.FloatTensor(y_val).unsqueeze(1).to(device)
            with autocast('cuda'):
                val_outputs = model(val_X)
                val_loss = criterion(val_outputs, val_y).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
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
    test_preds = (test_probs > 0.5).astype(int)
    
    auprc = average_precision_score(y_test, test_probs)
    auroc = roc_auc_score(y_test, test_probs)
    f1 = f1_score(y_test, test_preds)
    accuracy = accuracy_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'auprc': auprc,
        'auroc': auroc,
        'f1': f1,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

print("="*80)
print("RANDOM SEARCH TEST EVALUATION")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

X_train = np.load('data/X_train.npy')
X_val = np.load('data/X_val.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_val = np.load('data/y_val.npy')
y_test = np.load('data/y_test.npy')

random_configs = {
    'GRU': {
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'weight_decay': 0
    },
    'BiLSTM': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'weight_decay': 0
    },
    'CNN-BiLSTM': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_filters': 64,
        'kernel_size': 3,
        'weight_decay': 0
    }
}

results = {}

for model_name, config in random_configs.items():
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    
    if model_name == 'GRU':
        model = ConfigurableGRU(4, config['hidden_size'], config['num_layers'], config['dropout'])
    elif model_name == 'BiLSTM':
        model = ConfigurableBiLSTM(4, config['hidden_size'], config['num_layers'], config['dropout'])
    else:
        model = CNNBiLSTM4ch(4, config['hidden_size'], config['num_layers'], 
                            config['dropout'], config['num_filters'], config['kernel_size'])
    
    model = train_model(model, config, X_train, y_train, X_val, y_val)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    results[model_name] = metrics
    
    print(f"\nTest Results:")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']:6d}  FP: {metrics['fp']:6d}")
    print(f"  FN: {metrics['fn']:6d}  TP: {metrics['tp']:6d}")

os.makedirs('results/random_search_test', exist_ok=True)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'AUPRC': [results[m]['auprc'] for m in results.keys()],
    'AUROC': [results[m]['auroc'] for m in results.keys()],
    'F1': [results[m]['f1'] for m in results.keys()],
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'TN': [results[m]['tn'] for m in results.keys()],
    'FP': [results[m]['fp'] for m in results.keys()],
    'FN': [results[m]['fn'] for m in results.keys()],
    'TP': [results[m]['tp'] for m in results.keys()]
})

results_df.to_csv('results/random_search_test/random_search_test_metrics.csv', index=False)

print("\n" + "="*80)
print("RANDOM SEARCH TEST EVALUATION COMPLETE")
print("="*80)
print("\nResults saved: results/random_search_test/random_search_test_metrics.csv")