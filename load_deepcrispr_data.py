import pandas as pd
import numpy as np

# Paths to the off-target data files
data_files = {
    'hek293t': r'C:\Users\Pc\OneDrive\Desktop\CrisprAI\CRISPR_Model\data\paper_data\offtar\hek293t.epiotrt',
    'k562': r'C:\Users\Pc\OneDrive\Desktop\CrisprAI\CRISPR_Model\data\paper_data\offtar\k562.epiotrt'
}

all_data = []

for cell_line, file_path in data_files.items():
    print(f"\n{'='*60}")
    print(f"Loading: {cell_line}")
    print('='*60)
    
    # Read without header, then manually parse
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the data
    parsed_data = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            # Extract: sgRNA_id, sgRNA_seq, DNA_seq, ..., label
            # The format seems to be: sg_id, sequences..., label
            parsed_data.append(parts)
    
    # The last column should be the label (0 or 1)
    # Find which columns have actual sequences (23 nucleotides, not all N or A)
    print(f"Total rows: {len(parsed_data)}")
    print(f"Columns per row: {len(parsed_data[0])}")
    
    # Show first row to understand structure
    print(f"\nFirst row:")
    for i, val in enumerate(parsed_data[0]):
        print(f"  Col {i}: {val}")
    
    all_data.append((cell_line, parsed_data))

# Let's understand the structure better
print("\n" + "="*60)
print("Analyzing data structure...")
print("="*60)

# Check last column (should be label)
for cell_line, data in all_data:
    labels = [row[-1] for row in data[:100]]  # First 100 rows
    unique_labels = set(labels)
    print(f"\n{cell_line}:")
    print(f"  Unique values in last column: {unique_labels}")
    
    # Check for sequences (columns with ATCG)
    if len(data) > 0:
        for i in range(min(12, len(data[0]))):
            sample_val = data[0][i]
            if len(sample_val) == 23 and any(c in sample_val for c in 'ATCG'):
                print(f"  Column {i} looks like sequence: {sample_val}")