import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*60)
print("STEP 1: DATA PREPROCESSING")
print("="*60)

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load hek293t data
print("\nLoading hek293t cell line...")
hek293t_data = []
with open(r'data\paper_data\offtar\hek293t.epiotrt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 12:
            sgRNA = parts[1]
            DNA = parts[6]
            label = int(parts[11])
            if 'N' not in sgRNA and 'N' not in DNA:
                hek293t_data.append({
                    'sgRNA': sgRNA,
                    'DNA': DNA,
                    'label': label,
                    'cell_line': 'hek293t'
                })

print(f"  Loaded {len(hek293t_data):,} samples")

# Load k562 data
print("\nLoading k562 cell line...")
k562_data = []
with open(r'data\paper_data\offtar\k562.epiotrt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 12:
            sgRNA = parts[1]
            DNA = parts[6]
            label = int(parts[11])
            if 'N' not in sgRNA and 'N' not in DNA:
                k562_data.append({
                    'sgRNA': sgRNA,
                    'DNA': DNA,
                    'label': label,
                    'cell_line': 'k562'
                })

print(f"  Loaded {len(k562_data):,} samples")

# Combine datasets
df = pd.DataFrame(hek293t_data + k562_data)

print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"Total samples: {len(df):,}")
print(f"Unique sgRNAs: {df['sgRNA'].nunique()}")
print(f"\nCell line distribution:")
print(df['cell_line'].value_counts())
print(f"\nClass distribution:")
print(f"Positive (off-target): {(df['label']==1).sum():,} ({(df['label']==1).sum()/len(df)*100:.2f}%)")
print(f"Negative: {(df['label']==0).sum():,} ({(df['label']==0).sum()/len(df)*100:.2f}%)")

# Save clean dataset
df.to_csv('data/deepcrispr_clean.csv', index=False)
print(f"\nSaved: data/deepcrispr_clean.csv")

# VISUALIZATION 1: Dataset overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Class distribution pie chart
labels_count = df['label'].value_counts()
axes[0, 0].pie(labels_count, labels=['Negative (0)', 'Positive (1)'], 
               autopct='%1.2f%%', startangle=90, colors=['#3498db', '#e74c3c'])
axes[0, 0].set_title('Class Distribution', fontsize=12, fontweight='bold')

# Cell line distribution
cell_count = df['cell_line'].value_counts()
axes[0, 1].bar(cell_count.index, cell_count.values, color=['#2ecc71', '#9b59b6'], edgecolor='black')
axes[0, 1].set_ylabel('Number of Samples', fontweight='bold')
axes[0, 1].set_title('Cell Line Distribution', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(cell_count.values):
    axes[0, 1].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# Class distribution by cell line
cell_class = df.groupby(['cell_line', 'label']).size().unstack(fill_value=0)
cell_class.plot(kind='bar', ax=axes[1, 0], color=['#3498db', '#e74c3c'], edgecolor='black')
axes[1, 0].set_ylabel('Count', fontweight='bold')
axes[1, 0].set_xlabel('Cell Line', fontweight='bold')
axes[1, 0].set_title('Class Distribution by Cell Line', fontsize=12, fontweight='bold')
axes[1, 0].legend(['Negative', 'Positive'])
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
axes[1, 0].grid(axis='y', alpha=0.3)

# Statistics table
stats_data = [
    ['Total Samples', f'{len(df):,}'],
    ['HEK293T Samples', f'{(df["cell_line"]=="hek293t").sum():,}'],
    ['K562 Samples', f'{(df["cell_line"]=="k562").sum():,}'],
    ['Positive Samples', f'{(df["label"]==1).sum():,} ({(df["label"]==1).sum()/len(df)*100:.2f}%)'],
    ['Negative Samples', f'{(df["label"]==0).sum():,} ({(df["label"]==0).sum()/len(df)*100:.2f}%)'],
    ['Imbalance Ratio', f'1:{int((df["label"]==0).sum()/(df["label"]==1).sum())}']
]

axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=stats_data, colLabels=['Metric', 'Value'],
                          cellLoc='left', loc='center', colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(2):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(stats_data) + 1):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

axes[1, 1].set_title('Dataset Statistics', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/step1_dataset_overview.png', dpi=300, bbox_inches='tight')
print("Saved: figures/step1_dataset_overview.png")
plt.close()

# VISUALIZATION 2: Sample data table
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

sample_data = df.head(5).copy()
table_data = []
for idx, row in sample_data.iterrows():
    table_data.append([
        idx + 1,
        row['sgRNA'],
        row['DNA'],
        'Off-Target' if row['label'] == 1 else 'Non-Off-Target',
        row['cell_line'].upper()
    ])

table = ax.table(cellText=table_data,
                 colLabels=['No.', 'sgRNA (23nt)', 'DNA Target (23nt)', 'Label', 'Cell Line'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.08, 0.35, 0.35, 0.15, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(5):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, 6):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

plt.title('Sample Data (First 5 Rows)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('figures/step1_sample_data.png', dpi=300, bbox_inches='tight')
print("Saved: figures/step1_sample_data.png")
plt.close()

print("\n" + "="*60)
print("STEP 1 COMPLETE")
print("="*60)
print("\nGenerated 2 visualizations:")
print("  1. step1_dataset_overview.png")
print("  2. step1_sample_data.png")