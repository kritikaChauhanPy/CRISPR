"""
CNN-BiLSTM-4ch Genetic Algorithm - Optimized for Improvement
This script implements a genetic algorithm to optimize the hyperparameters of a CNN-BiLSTM model
using 4-channel one-hot encoded CRISPR data. The goal is to maximize the validation AUPRC.
The genetic algorithm includes:
1. Individual creation with adaptive constraints based on hidden size.
2. Tournament selection for parent selection.
3. Crossover and mutation operations with adaptive constraints.
Model architecture:
- Convolutional layer with ReLU and MaxPooling
- Bidirectional LSTM layers
- Fully connected output layer
- Mixed precision training for efficiency
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scipy.special import expit
import random
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CNNBiLSTM4ch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_filters=64, kernel_size=3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, 
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_and_evaluate(model_class, config, X_train, X_val, y_train, y_val):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = model_class(
            input_size=4,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_filters=config.get('num_filters', 64),
            kernel_size=config.get('kernel_size', 3)
        ).to(device)
        
        pos_count = sum(y_train)
        neg_count = len(y_train) - pos_count
        pos_weight = torch.tensor([neg_count / pos_count]).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        from torch.amp import autocast, GradScaler
        scaler = GradScaler('cuda')
        
        best_val_auprc = 0
        patience = 10
        patience_counter = 0
        max_epochs = 50
        
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
                
                if i % (config['batch_size'] * 10) == 0:
                    torch.cuda.empty_cache()
            
            model.eval()
            with torch.no_grad():
                val_X = torch.FloatTensor(X_val).to(device)
                with autocast('cuda'):
                    val_logits = model(val_X).cpu().numpy().flatten()
                val_probs = expit(val_logits)
                val_auprc = average_precision_score(y_val, val_probs)
            
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        del model, optimizer, criterion
        torch.cuda.empty_cache()
        
        return best_val_auprc
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"    Memory error - assigning penalty score")
            torch.cuda.empty_cache()
            return 0.50
        else:
            raise e

def create_individual():
    hidden = random.choice([96, 128, 160, 192])
    
    if hidden >= 160:
        batch = random.choice([32, 64])
        filters = random.choice([32, 64])
        layers = 1
    else:
        batch = random.choice([64, 128])
        filters = random.choice([64, 96])
        layers = random.choice([1, 2])
    
    return {
        'hidden_size': hidden,
        'num_layers': layers,
        'dropout': round(random.uniform(0.15, 0.48), 2),
        'learning_rate': random.choice([0.0005, 0.001]),
        'batch_size': batch,
        'num_filters': filters,
        'kernel_size': random.choice([3, 5])
    }

def tournament_selection(population, fitness_scores, tournament_size=3):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_index].copy()

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    
    if child['hidden_size'] >= 160:
        child['batch_size'] = min(child['batch_size'], 64)
        child['num_filters'] = min(child['num_filters'], 64)
        child['num_layers'] = 1
    
    return child

def mutate(individual, mutation_prob=0.2):
    mutated = individual.copy()
    
    if random.random() < mutation_prob:
        mutated['hidden_size'] = random.choice([96, 128, 160, 192])
    
    if random.random() < mutation_prob:
        mutated['num_layers'] = random.choice([1, 2]) if mutated['hidden_size'] < 160 else 1
    
    if random.random() < mutation_prob:
        mutated['dropout'] = round(random.uniform(0.15, 0.48), 2)
    
    if random.random() < mutation_prob:
        mutated['learning_rate'] = random.choice([0.0005, 0.001])
    
    if random.random() < mutation_prob:
        max_batch = 64 if mutated['hidden_size'] >= 160 else 128
        mutated['batch_size'] = random.choice([32, 64, max_batch])
    
    if random.random() < mutation_prob:
        max_filters = 64 if mutated['hidden_size'] >= 160 else 96
        mutated['num_filters'] = random.choice([32, 64, max_filters])
    
    if random.random() < mutation_prob:
        mutated['kernel_size'] = random.choice([3, 5])
    
    return mutated

def run_genetic_algorithm(model_class, model_name, X_train, X_val, y_train, y_val,
                         population_size=20, num_generations=5, elite_size=4,
                         crossover_prob=0.3, mutation_prob=0.2):
    
    print(f"\n{'='*80}")
    print(f"GENETIC ALGORITHM: {model_name}")
    print(f"{'='*80}")
    print(f"Population: {population_size}")
    print(f"Generations: {num_generations}")
    print(f"Elites: {elite_size}")
    print(f"Crossover prob: {crossover_prob}")
    print(f"Mutation prob: {mutation_prob}")
    
    results = []
    population = []
    fitness_scores = []
    
    print(f"\n{'-'*80}")
    print(f"GENERATION 1: Random Initialization")
    print(f"{'-'*80}\n")
    
    for i in range(population_size):
        individual = create_individual()
        print(f"Individual {i+1}/{population_size}")
        print(f"  Config: {individual}")
        
        auprc = train_and_evaluate(model_class, individual, X_train, X_val, y_train, y_val)
        print(f"  AUPRC: {auprc:.4f}\n")
        
        population.append(individual)
        fitness_scores.append(auprc)
        
        results.append({
            'generation': 1,
            'individual': i + 1,
            **individual,
            'val_auprc': auprc
        })
    
    best_gen1 = max(fitness_scores)
    print(f"Generation 1 Best: {best_gen1:.4f}")
    
    for gen in range(2, num_generations + 1):
        print(f"\n{'-'*80}")
        print(f"GENERATION {gen}")
        print(f"{'-'*80}\n")
        
        elite_indices = sorted(range(len(fitness_scores)), 
                             key=lambda i: fitness_scores[i], 
                             reverse=True)[:elite_size]
        
        print("Elite configurations:")
        for rank, idx in enumerate(elite_indices, 1):
            print(f"  Elite {rank}: AUPRC {fitness_scores[idx]:.4f}")
            print(f"    Config: {population[idx]}")
        
        new_population = [population[i].copy() for i in elite_indices]
        new_fitness = [fitness_scores[i] for i in elite_indices]
        
        offspring_count = population_size - elite_size
        
        for i in range(offspring_count):
            if random.random() < crossover_prob:
                parent1 = tournament_selection(population, fitness_scores)
                parent2 = tournament_selection(population, fitness_scores)
                child = crossover(parent1, parent2)
            else:
                child = tournament_selection(population, fitness_scores)
            
            child = mutate(child, mutation_prob)
            
            print(f"\nOffspring {i+1}/{offspring_count}")
            print(f"  Config: {child}")
            
            auprc = train_and_evaluate(model_class, child, X_train, X_val, y_train, y_val)
            print(f"  AUPRC: {auprc:.4f}")
            
            new_population.append(child)
            new_fitness.append(auprc)
            
            results.append({
                'generation': gen,
                'individual': elite_size + i + 1,
                **child,
                'val_auprc': auprc
            })
        
        population = new_population
        fitness_scores = new_fitness
        
        gen_best = max(fitness_scores)
        gen_avg = np.mean(fitness_scores)
        improvement = gen_best - best_gen1
        
        print(f"\nGeneration {gen} Summary:")
        print(f"  Best: {gen_best:.4f}")
        print(f"  Average: {gen_avg:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
    
    best_idx = fitness_scores.index(max(fitness_scores))
    best_config = population[best_idx]
    best_auprc = fitness_scores[best_idx]
    
    results_df = pd.DataFrame(results)
    
    return best_config, best_auprc, results_df

def create_visualizations(results_df, model_name, best_auprc):
    
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    sns.set_style("whitegrid")
    output_dir = f'results/true_ga/{model_name}_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    generations = results_df.groupby('generation')['val_auprc'].agg(['max', 'mean', 'min'])
    
    plt.plot(generations.index, generations['max'], 'o-', 
             color='green', linewidth=2, markersize=8, label='Best')
    plt.plot(generations.index, generations['mean'], 's-', 
             color='blue', linewidth=2, markersize=6, label='Average')
    plt.plot(generations.index, generations['min'], '^-', 
             color='red', linewidth=2, markersize=6, label='Worst')
    
    plt.xlabel('Generation', fontsize=14, fontweight='bold')
    plt.ylabel('Validation AUPRC', fontsize=14, fontweight='bold')
    plt.title(f'{model_name} - GA Convergence', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 6))
    
    best_gen = results_df.loc[results_df['val_auprc'].idxmax(), 'generation']
    plt.annotate(f'Best: {best_auprc:.4f}', 
                xy=(best_gen, best_auprc),
                xytext=(best_gen+0.5, best_auprc-0.02),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_convergence.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: 1_convergence.png")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(results_df['val_auprc'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(results_df['val_auprc'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["val_auprc"].mean():.4f}')
    plt.axvline(best_auprc, color='green', linestyle='--', 
                linewidth=2, label=f'Best: {best_auprc:.4f}')
    plt.xlabel('Validation AUPRC', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('AUPRC Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    box_data = [results_df[results_df['generation']==i]['val_auprc'].values for i in range(1, 6)]
    bp = plt.boxplot(box_data, labels=[f'Gen {i}' for i in range(1, 6)], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    plt.ylabel('Validation AUPRC', fontsize=12, fontweight='bold')
    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.title('Distribution by Generation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: 2_distribution.png")
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    axes[0, 0].scatter(results_df['hidden_size'], results_df['val_auprc'], alpha=0.6, s=50)
    axes[0, 0].set_xlabel('Hidden Size', fontweight='bold')
    axes[0, 0].set_ylabel('Validation AUPRC', fontweight='bold')
    axes[0, 0].set_title('Hidden Size Impact')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(results_df['dropout'], results_df['val_auprc'], alpha=0.6, s=50, color='orange')
    axes[0, 1].set_xlabel('Dropout', fontweight='bold')
    axes[0, 1].set_ylabel('Validation AUPRC', fontweight='bold')
    axes[0, 1].set_title('Dropout Impact')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].scatter(results_df['learning_rate'], results_df['val_auprc'], alpha=0.6, s=50, color='green')
    axes[0, 2].set_xlabel('Learning Rate', fontweight='bold')
    axes[0, 2].set_ylabel('Validation AUPRC', fontweight='bold')
    axes[0, 2].set_title('Learning Rate Impact')
    axes[0, 2].set_xscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[0, 3].scatter(results_df['batch_size'], results_df['val_auprc'], alpha=0.6, s=50, color='red')
    axes[0, 3].set_xlabel('Batch Size', fontweight='bold')
    axes[0, 3].set_ylabel('Validation AUPRC', fontweight='bold')
    axes[0, 3].set_title('Batch Size Impact')
    axes[0, 3].grid(True, alpha=0.3)
    
    axes[1, 0].boxplot([results_df[results_df['num_layers']==1]['val_auprc'].values,
                        results_df[results_df['num_layers']==2]['val_auprc'].values],
                       labels=['1 Layer', '2 Layers'])
    axes[1, 0].set_ylabel('Validation AUPRC', fontweight='bold')
    axes[1, 0].set_title('Number of Layers Impact')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].scatter(results_df['num_filters'], results_df['val_auprc'], alpha=0.6, s=50, color='purple')
    axes[1, 1].set_xlabel('CNN Filters', fontweight='bold')
    axes[1, 1].set_ylabel('Validation AUPRC', fontweight='bold')
    axes[1, 1].set_title('CNN Filters Impact')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(results_df['kernel_size'], results_df['val_auprc'], alpha=0.6, s=50, color='teal')
    axes[1, 2].set_xlabel('Kernel Size', fontweight='bold')
    axes[1, 2].set_ylabel('Validation AUPRC', fontweight='bold')
    axes[1, 2].set_title('Kernel Size Impact')
    axes[1, 2].grid(True, alpha=0.3)
    
    top_10 = results_df.nlargest(10, 'val_auprc')
    axes[1, 3].barh(range(10), top_10['val_auprc'].values, color='brown', alpha=0.7)
    axes[1, 3].set_yticks(range(10))
    axes[1, 3].set_yticklabels([f"Config {i+1}" for i in range(10)])
    axes[1, 3].set_xlabel('Validation AUPRC', fontweight='bold')
    axes[1, 3].set_title('Top 10 Configurations')
    axes[1, 3].grid(True, alpha=0.3, axis='x')
    axes[1, 3].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_hyperparameter_impact.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: 3_hyperparameter_impact.png")
    
    plt.figure(figsize=(10, 6))
    
    benchmarks = {
        'CNN-BiLSTM\nRandom': 0.6213,
        'Rahman\nLSTM': 0.7427,
        f'{model_name}\nGA': best_auprc
    }
    
    colors_list = ['red', 'gray', 'green']
    bars = plt.bar(benchmarks.keys(), benchmarks.values(), color=colors_list, 
                   edgecolor='black', linewidth=2, alpha=0.7)
    
    for bar, (name, value) in zip(bars, benchmarks.items()):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Validation AUPRC', fontsize=14, fontweight='bold')
    plt.title(f'{model_name} - Benchmark Comparison', fontsize=16, fontweight='bold')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: 4_benchmark_comparison.png")
    
    with open(f'{output_dir}/SUMMARY.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{model_name} GENETIC ALGORITHM - VALIDATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION SPACE:\n")
        f.write("-"*80 + "\n")
        f.write("Hidden size: 96-192 (adaptive)\n")
        f.write("Layers: 1-2 (constrained by hidden size)\n")
        f.write("Batch size: 32-128 (adaptive)\n")
        f.write("CNN filters: 32-96 (adaptive)\n")
        f.write("Kernel size: 3-5\n")
        f.write("Dropout: 0.15-0.48\n")
        f.write("Learning rate: 0.0005-0.001\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total configs: {len(results_df)}\n")
        f.write(f"Best AUPRC: {best_auprc:.4f}\n")
        f.write(f"Average AUPRC: {results_df['val_auprc'].mean():.4f}\n")
        f.write(f"Std Dev: {results_df['val_auprc'].std():.4f}\n\n")
        
        best = results_df.loc[results_df['val_auprc'].idxmax()]
        f.write("BEST CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        f.write(f"Generation: {int(best['generation'])}\n")
        f.write(f"Individual: {int(best['individual'])}\n")
        f.write(f"Hidden size: {int(best['hidden_size'])}\n")
        f.write(f"Layers: {int(best['num_layers'])}\n")
        f.write(f"Dropout: {best['dropout']:.2f}\n")
        f.write(f"Learning rate: {best['learning_rate']:.4f}\n")
        f.write(f"Batch size: {int(best['batch_size'])}\n")
        f.write(f"CNN filters: {int(best['num_filters'])}\n")
        f.write(f"Kernel size: {int(best['kernel_size'])}\n")
        f.write(f"Validation AUPRC: {best['val_auprc']:.4f}\n\n")
        
        f.write("COMPARISON:\n")
        f.write("-"*80 + "\n")
        f.write(f"Random search: 0.6213\n")
        f.write(f"GA optimized: {best_auprc:.4f}\n")
        improvement = ((best_auprc - 0.6213) / 0.6213) * 100
        f.write(f"Improvement: +{improvement:.1f}%\n\n")
        
        f.write("NOTE: Validation results only. Test set evaluation required.\n")
    
    print(f"  Saved: SUMMARY.txt")
    print(f"\n  All visualizations: {output_dir}/")

if __name__ == "__main__":
    
    print("="*80)
    print("CNN-BiLSTM-4ch GENETIC ALGORITHM")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\nLoading data...")
    X_train = np.load('data/X_train.npy')
    X_val = np.load('data/X_val.npy')
    y_train = np.load('data/y_train.npy')
    y_val = np.load('data/y_val.npy')
    
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    
    start_time = time.time()
    
    best_config, best_auprc, results_df = run_genetic_algorithm(
        model_class=CNNBiLSTM4ch,
        model_name='CNN-BiLSTM-4ch',
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        population_size=20,
        num_generations=5,
        elite_size=4,
        crossover_prob=0.3,
        mutation_prob=0.2
    )
    
    elapsed_time = time.time() - start_time
    
    os.makedirs('results/true_ga', exist_ok=True)
    results_df.to_csv('results/true_ga/CNN-BiLSTM-4ch_progress.csv', index=False)
    
    print(f"\n{'='*80}")
    print("CNN-BiLSTM-4ch COMPLETE")
    print(f"{'='*80}\n")
    
    print("Best Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nValidation AUPRC: {best_auprc:.4f}")
    print(f"Time: {elapsed_time/3600:.2f} hours")
    print(f"Total configs: {len(results_df)}")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*80}")
    
    benchmarks = [
        ('CNN-BiLSTM Random Search', 0.6213),
        ('Rahman LSTM Validation', 0.7427),
        ('CNN-BiLSTM-4ch GA', best_auprc)
    ]
    
    for name, score in benchmarks:
        print(f"{name:30s}: {score:.4f}")
    
    improvement = ((best_auprc - 0.6213) / 0.6213) * 100
    print(f"\nImprovement vs random: +{improvement:.1f}%")
    
    if best_auprc > 0.7427:
        print(f"Beats Rahman validation")
    
    create_visualizations(results_df, 'CNN-BiLSTM-4ch', best_auprc)
    
    print(f"\n{'='*80}")
    print("ALL COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults: results/true_ga/CNN-BiLSTM-4ch_progress.csv")
    print(f"Visualizations: results/true_ga/CNN-BiLSTM-4ch_visualizations/")
    print(f"\nNext: Test on test set for final evaluation")