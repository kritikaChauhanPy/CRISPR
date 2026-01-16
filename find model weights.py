import os

print("Searching for PyTorch model files (.pth, .pt, .pkl)...")
print("="*80)

search_dirs = [
    'results',
    'models',
    'checkpoints',
    'saved_models',
    'weights',
    '.'
]

found_files = []

for search_dir in search_dirs:
    if os.path.exists(search_dir):
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith(('.pth', '.pt', '.pkl')):
                    full_path = os.path.join(root, file)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    found_files.append((full_path, size_mb))

if found_files:
    print(f"Found {len(found_files)} model file(s):")
    print("-"*80)
    for path, size in found_files:
        print(f"{path:<60s} ({size:.2f} MB)")
else:
    print("NO MODEL WEIGHT FILES FOUND")
    print("\nThis means your step5 code did NOT save trained models.")
    print("You need to either:")
    print("  1. Modify step5 to save weights and rerun")
    print("  2. Continue with current approach (retrain each time)")

print("="*80)