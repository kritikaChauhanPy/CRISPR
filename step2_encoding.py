import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

print("="*60)
print("STEP 2: ONE-HOT ENCODING (4-Channel)")
print("="*60)

os.makedirs('figures', exist_ok=True)

# Load data
df = pd.read_csv('data/deepcrispr_clean.csv')
print(f"\nLoaded {len(df):,} samples")
print(f"Cell lines: {df['cell_line'].unique()}")

# One-hot encoding function
def one_hot_encode(sequence):
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, 4), dtype=np.float32)
    
    for i, nucleotide in enumerate(sequence):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1.0
    
    return one_hot

# Encode pair with logical OR
def encode_pair(sgRNA, DNA):
    sgRNA_encoded = one_hot_encode(sgRNA)
    DNA_encoded = one_hot_encode(DNA)
    combined = np.logical_or(sgRNA_encoded, DNA_encoded).astype(np.float32)
    return combined

# VISUALIZATION: Encoding example
print("\nCreating encoding visualization...")

sample_sgRNA = df.iloc[0]['sgRNA']
sample_DNA = df.iloc[0]['DNA']
sample_cell = df.iloc[0]['cell_line']

sgRNA_encoded = one_hot_encode(sample_sgRNA)
DNA_encoded = one_hot_encode(sample_DNA)
combined_encoded = encode_pair(sample_sgRNA, sample_DNA)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

im1 = axes[0].imshow(sgRNA_encoded.T, cmap='Blues', aspect='auto')
axes[0].set_yticks([0, 1, 2, 3])
axes[0].set_yticklabels(['A', 'T', 'C', 'G'])
axes[0].set_title(f'sgRNA: {sample_sgRNA} (Cell Line: {sample_cell.upper()})', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Channel')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(DNA_encoded.T, cmap='Greens', aspect='auto')
axes[1].set_yticks([0, 1, 2, 3])
axes[1].set_yticklabels(['A', 'T', 'C', 'G'])
axes[1].set_title(f'DNA: {sample_DNA}', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Channel')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(combined_encoded.T, cmap='Reds', aspect='auto')
axes[2].set_yticks([0, 1, 2, 3])
axes[2].set_yticklabels(['A', 'T', 'C', 'G'])
axes[2].set_title('Combined (Logical OR)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Sequence Position (0-22)')
axes[2].set_ylabel('Channel')
plt.colorbar(im3, ax=axes[2])

plt.suptitle('4-Channel One-Hot Encoding Process', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/step2_encoding.png', dpi=300, bbox_inches='tight')
print("Saved: figures/step2_encoding.png")
plt.close()

# Encode all sequences
print("\nEncoding all sequences...")
X = []
y = []
cell_lines = []

for idx, row in df.iterrows():
    if idx % 10000 == 0:
        print(f"  Processed {idx:,}/{len(df):,}...")
    
    encoded = encode_pair(row['sgRNA'], row['DNA'])
    X.append(encoded)
    y.append(row['label'])
    cell_lines.append(row['cell_line'])

X = np.array(X)
y = np.array(y)
cell_lines = np.array(cell_lines)

print(f"\nEncoding complete!")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Stratified split by both label AND cell line
print("\nSplitting data (stratified by label and cell line)...")

df['strat_column'] = df['label'].astype(str) + '_' + df['cell_line']

X_temp, X_test, y_temp, y_test, cell_temp, cell_test, idx_temp, idx_test = train_test_split(
    X, y, cell_lines, df.index.values,
    test_size=0.1,
    random_state=42,
    stratify=df['strat_column']
)

strat_temp = df.loc[idx_temp, 'strat_column'].values
X_train, X_val, y_train, y_val, cell_train, cell_val = train_test_split(
    X_temp, y_temp, cell_temp,
    test_size=0.1111,
    random_state=42,
    stratify=strat_temp
)

print(f"\nTrain: {len(X_train):,} samples ({np.sum(y_train==1)} positive)")
print(f"  HEK293T: {np.sum(cell_train=='hek293t'):,}, K562: {np.sum(cell_train=='k562'):,}")
print(f"Val:   {len(X_val):,} samples ({np.sum(y_val==1)} positive)")
print(f"  HEK293T: {np.sum(cell_val=='hek293t'):,}, K562: {np.sum(cell_val=='k562'):,}")
print(f"Test:  {len(X_test):,} samples ({np.sum(y_test==1)} positive)")
print(f"  HEK293T: {np.sum(cell_test=='hek293t'):,}, K562: {np.sum(cell_test=='k562'):,}")

pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"\nClass weight: {pos_weight:.2f}")

# VISUALIZATION: Data splits
fig, ax = plt.subplots(figsize=(10, 6))

splits = ['Train\n(80%)', 'Val\n(10%)', 'Test\n(10%)']
sizes = [len(X_train), len(X_val), len(X_test)]
pos_samples = [np.sum(y_train==1), np.sum(y_val==1), np.sum(y_test==1)]

x = np.arange(len(splits))
width = 0.35

bars1 = ax.bar(x - width/2, sizes, width, label='Total Samples', color='#3498db')
bars2 = ax.bar(x + width/2, pos_samples, width, label='Positive Samples', color='#e74c3c')

ax.set_ylabel('Number of Samples')
ax.set_title('Train/Validation/Test Split Distribution', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/step2_splits.png', dpi=300, bbox_inches='tight')
print("Saved: figures/step2_splits.png")
plt.close()

# Save data
print("\nSaving preprocessed data...")
np.save('data/X_train.npy', X_train)
np.save('data/X_val.npy', X_val)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_val.npy', y_val)
np.save('data/y_test.npy', y_test)
np.save('data/cell_train.npy', cell_train)
np.save('data/cell_val.npy', cell_val)
np.save('data/cell_test.npy', cell_test)

with open('data/pos_weight.txt', 'w') as f:
    f.write(str(pos_weight))

print("\n" + "="*60)
print("STEP 2 COMPLETE")
print("="*60)
print("\nGenerated 2 visualizations:")
print("  1. step2_encoding.png")
print("  2. step2_splits.png")