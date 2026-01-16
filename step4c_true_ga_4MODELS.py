"""
step4c_TRUE_genetic_algorithm_4MODELS.py

TRUE Genetic Algorithm with Crossover and Mutation (Rahman-style)
Optimized for 4 models only:
1. GRU-4ch (your champion)
2. BiLSTM-4ch (failure case validation)
3. CNN-BiLSTM-4ch (baseline)
4. CNN-BiLSTM-5ch (novel contribution)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
import random
import os
import time
from datetime import datetime

print("="*80)
print("TRUE GENETIC ALGORITHM - 4 MODEL OPTIMIZATION")
print("With Crossover and Mutation (Rahman-style)")
print("="*80)
print("\nModels to optimize:")
print("1. GRU-4ch")
print("2. BiLSTM-4ch")
print("3. CNN-BiLSTM-4ch")
print("4. CNN-BiLSTM-5ch")
print("\nTotal estimated time: ~5.5 days")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load data
print("\nLoading data...")
X_train_4ch = np.load('data/X_train.npy')
X_train_5ch = np.load('data/X_train_5ch.npy')
y_train = np.load('data/y_train.npy')
X_val_4ch = np.load('data/X_val.npy')
X_val_5ch = np.load('data/X_val_5ch.npy')
y_val = np.load('data/y_val.npy')

print(f"Training: {len(X_train_4ch)} samples")
print(f"Validation: {len(X_val_4ch)} samples")
print(f"4-channel shape: {X_train_4ch.shape}")
print(f"5-channel shape: {X_train_5ch.shape}")

# Create directories
os.makedirs('models/true_ga', exist_ok=True)
os.makedirs('results/true_ga', exist_ok=True)

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class ConfigurableGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3):
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
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3):
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


class ConfigurableCNNBiLSTM(nn.Module):
    def __init__(self, input_channels=4, cnn_filters=128, lstm_hidden=64, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, cnn_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(input_channels, cnn_filters, 5, padding=2)
        self.conv3 = nn.Conv1d(input_channels, cnn_filters, 7, padding=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(cnn_filters * 3, lstm_hidden, num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([
            self.relu(self.conv1(x)),
            self.relu(self.conv2(x)),
            self.relu(self.conv3(x))
        ], dim=1)
        x = self.dropout(x).transpose(1, 2)
        bilstm_out, _ = self.bilstm(x)
        x = self.relu(self.fc1(bilstm_out[:, -1, :]))
        return self.fc2(self.dropout(x))


# =============================================================================
# GENETIC ALGORITHM FUNCTIONS
# =============================================================================

def create_random_individual(model_type):
    """Create random hyperparameter configuration"""
    individual = {
        'hidden_size': random.choice([64, 96, 128, 160, 192, 256]),
        'num_layers': random.choice([1, 2]),
        'dropout': round(random.uniform(0.1, 0.5), 2),
        'learning_rate': random.choice([1e-4, 5e-4, 1e-3]),
        'batch_size': random.choice([32, 64, 128])
    }
    
    if model_type == 'CNN-BiLSTM':
        individual['cnn_filters'] = random.choice([64, 128, 256])
        individual['lstm_hidden'] = random.choice([32, 64, 128])
    
    return individual


def tournament_selection(population, fitness, tournament_size=2):
    """Rahman-style tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx].copy()


def crossover(parent1, parent2, crossover_prob=0.3):
    """Rahman-style crossover with probability"""
    child = {}
    
    for param in parent1.keys():
        if random.random() < crossover_prob:
            child[param] = parent2[param]
        else:
            child[param] = parent1[param]
    
    return child


def mutate(individual, mutation_prob=0.2, model_type='GRU'):
    """Rahman-style mutation with probability"""
    mutated = individual.copy()
    
    for param in mutated.keys():
        if random.random() < mutation_prob:
            if param == 'hidden_size':
                mutated[param] = random.choice([64, 96, 128, 160, 192, 256])
            elif param == 'num_layers':
                mutated[param] = random.choice([1, 2])
            elif param == 'dropout':
                mutated[param] = round(random.uniform(0.1, 0.5), 2)
            elif param == 'learning_rate':
                mutated[param] = random.choice([1e-4, 5e-4, 1e-3])
            elif param == 'batch_size':
                mutated[param] = random.choice([32, 64, 128])
            elif param == 'cnn_filters':
                mutated[param] = random.choice([64, 128, 256])
            elif param == 'lstm_hidden':
                mutated[param] = random.choice([32, 64, 128])
    
    return mutated


def train_and_evaluate(model_class, hyperparams, X_train, X_val, y_train, y_val, 
                      max_epochs=50, patience=10):
    """Train model and return validation AUPRC"""
    
    if model_class == ConfigurableGRU or model_class == ConfigurableBiLSTM:
        model = model_class(
            input_size=X_train.shape[2],
            hidden_size=hyperparams['hidden_size'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout']
        ).to(device)
    else:
        model = model_class(
            input_channels=X_train.shape[2],
            cnn_filters=hyperparams.get('cnn_filters', 128),
            lstm_hidden=hyperparams.get('lstm_hidden', 64),
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout']
        ).to(device)
    
    pos_weight = torch.tensor([232.94]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    
    best_val_auprc = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_X = torch.FloatTensor(X_val).to(device)
            val_logits = model(val_X).cpu().numpy().flatten()
            val_probs = 1 / (1 + np.exp(-val_logits))
            val_auprc = average_precision_score(y_val, val_probs)
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_auprc


def run_genetic_algorithm(
    model_class,
    model_name,
    X_train,
    X_val,
    y_train,
    y_val,
    population_size=20,
    num_generations=5,
    elite_size=4,
    crossover_prob=0.3,
    mutation_prob=0.2
):
    """TRUE Genetic Algorithm with Crossover and Mutation"""
    
    print("\n" + "="*80)
    print(f"GENETIC ALGORITHM: {model_name}")
    print("="*80)
    print(f"Population: {population_size}")
    print(f"Generations: {num_generations}")
    print(f"Elites: {elite_size}")
    print(f"Crossover prob: {crossover_prob}")
    print(f"Mutation prob: {mutation_prob}")
    
    model_type = 'CNN-BiLSTM' if 'CNN' in model_name else 'GRU'
    
    start_time = time.time()
    all_results = []
    
    print("\n" + "-"*80)
    print("GENERATION 1: Random Initialization")
    print("-"*80)
    
    population = [create_random_individual(model_type) for _ in range(population_size)]
    fitness = []
    
    for idx, individual in enumerate(population):
        print(f"\nIndividual {idx+1}/{population_size}")
        print(f"  Config: {individual}")
        
        auprc = train_and_evaluate(model_class, individual, X_train, X_val, y_train, y_val)
        fitness.append(auprc)
        
        print(f"  AUPRC: {auprc:.4f}")
        
        all_results.append({
            'model': model_name,
            'generation': 1,
            'individual': idx,
            'source': 'random',
            'hidden_size': individual.get('hidden_size', 0),
            'num_layers': individual.get('num_layers', 0),
            'dropout': individual.get('dropout', 0),
            'learning_rate': individual.get('learning_rate', 0),
            'batch_size': individual.get('batch_size', 0),
            'cnn_filters': individual.get('cnn_filters', 0),
            'lstm_hidden': individual.get('lstm_hidden', 0),
            'val_auprc': auprc
        })
    
    best_gen1 = max(fitness)
    print(f"\nGeneration 1 Best: {best_gen1:.4f}")
    
    for generation in range(2, num_generations + 1):
        print("\n" + "-"*80)
        print(f"GENERATION {generation}")
        print("-"*80)
        
        elite_indices = np.argsort(fitness)[-elite_size:]
        elites = [population[i] for i in elite_indices]
        elite_fitness = [fitness[i] for i in elite_indices]
        
        print(f"\nElite configurations:")
        for i, (elite, fit) in enumerate(zip(elites, elite_fitness)):
            print(f"  Elite {i+1}: AUPRC {fit:.4f}")
            print(f"    Config: {elite}")
        
        new_population = elites.copy()
        new_fitness = elite_fitness.copy()
        
        offspring_needed = population_size - elite_size
        
        for i in range(offspring_needed):
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            
            child = crossover(parent1, parent2, crossover_prob)
            child = mutate(child, mutation_prob, model_type)
            
            print(f"\nOffspring {i+1}/{offspring_needed}")
            print(f"  Config: {child}")
            
            auprc = train_and_evaluate(model_class, child, X_train, X_val, y_train, y_val)
            print(f"  AUPRC: {auprc:.4f}")
            
            new_population.append(child)
            new_fitness.append(auprc)
            
            all_results.append({
                'model': model_name,
                'generation': generation,
                'individual': elite_size + i,
                'source': 'crossover+mutation',
                'hidden_size': child.get('hidden_size', 0),
                'num_layers': child.get('num_layers', 0),
                'dropout': child.get('dropout', 0),
                'learning_rate': child.get('learning_rate', 0),
                'batch_size': child.get('batch_size', 0),
                'cnn_filters': child.get('cnn_filters', 0),
                'lstm_hidden': child.get('lstm_hidden', 0),
                'val_auprc': auprc
            })
        
        population = new_population
        fitness = new_fitness
        
        best_gen = max(fitness)
        avg_gen = np.mean(fitness)
        
        print(f"\nGeneration {generation} Summary:")
        print(f"  Best: {best_gen:.4f}")
        print(f"  Average: {avg_gen:.4f}")
        print(f"  Improvement from Gen 1: {(best_gen - best_gen1):.4f}")
        
        df_progress = pd.DataFrame(all_results)
        df_progress.to_csv(f'results/true_ga/{model_name}_progress.csv', index=False)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"{model_name} COMPLETE")
    print("="*80)
    
    best_idx = np.argmax(fitness)
    best_individual = population[best_idx]
    best_auprc = fitness[best_idx]
    
    print(f"\nBest Configuration:")
    for key, value in best_individual.items():
        print(f"  {key}: {value}")
    print(f"\nValidation AUPRC: {best_auprc:.4f}")
    print(f"Improvement from Gen 1: {(best_auprc - best_gen1):.4f}")
    print(f"\nTime taken: {elapsed/3600:.2f} hours")
    print(f"Total configs tested: {len(all_results)}")
    
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(f'results/true_ga/{model_name}_final.csv', index=False)
    
    with open(f'results/true_ga/{model_name}_best_config.txt', 'w') as f:
        f.write(f"Best Configuration for {model_name}\n")
        f.write("="*50 + "\n")
        for key, value in best_individual.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nValidation AUPRC: {best_auprc:.4f}\n")
        f.write(f"Improvement from Gen 1: {(best_auprc - best_gen1):.4f}\n")
        f.write(f"Time: {elapsed/3600:.2f} hours\n")
        f.write(f"Total configs tested: {len(all_results)}\n")
    
    return best_individual, best_auprc, df_final


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("EXECUTION PLAN - 4 MODELS")
    print("="*80)
    print("\n4 GA runs to execute:")
    print("1. GRU-4ch (est. 1.1 days)")
    print("2. BiLSTM-4ch (est. 1.3 days)")  
    print("3. CNN-BiLSTM-4ch (est. 1.5 days)")
    print("4. CNN-BiLSTM-5ch (est. 1.5 days)")
    print("\nTotal estimated time: ~5.5 days")
    
    print("\n" + "="*80)
    print("CHECKPOINT AFTER RUN 2")
    print("="*80)
    print("After BiLSTM-4ch completes (~2.5 days), you will be asked:")
    print("  'Continue with CNN models? (yes/no)'")
    print("\nThis allows you to:")
    print("  - Check if BiLSTM improved (or still failed)")
    print("  - Decide whether to continue")
    print("  - Stop early if needed")
    print("="*80)
    
    response = input("\nStart TRUE GA training? (yes/no): ")
    
    if response.lower() != 'yes':
        print("\nStopped. Use current results and write thesis honestly.")
        exit(0)
    
    print(f"\nStarting at: {datetime.now()}")
    
    all_results_summary = []
    
    runs = [
        ('GRU-4ch', ConfigurableGRU, X_train_4ch, X_val_4ch),
        ('BiLSTM-4ch', ConfigurableBiLSTM, X_train_4ch, X_val_4ch),
        ('CNN-BiLSTM-4ch', ConfigurableCNNBiLSTM, X_train_4ch, X_val_4ch),
        ('CNN-BiLSTM-5ch', ConfigurableCNNBiLSTM, X_train_5ch, X_val_5ch),
    ]
    
    for run_idx, (name, model_class, X_tr, X_v) in enumerate(runs, 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_idx}/4: {name}")
        print(f"{'='*80}")
        
        best_config, best_auprc, results_df = run_genetic_algorithm(
            model_class=model_class,
            model_name=name,
            X_train=X_tr,
            X_val=X_v,
            y_train=y_train,
            y_val=y_val,
            population_size=20,
            num_generations=5,
            elite_size=4,
            crossover_prob=0.3,
            mutation_prob=0.2
        )
        
        all_results_summary.append({
            'Model': name,
            'Best_AUPRC': best_auprc,
            'Best_Config': str(best_config)
        })
        
        if run_idx == 2:
            print("\n" + "="*80)
            print("CHECKPOINT: 2 RUNS COMPLETE (~2.5 days elapsed)")
            print("="*80)
            print("\nResults so far:")
            for r in all_results_summary:
                print(f"  {r['Model']:20s}: AUPRC {r['Best_AUPRC']:.4f}")
            
            print("\nAnalysis:")
            gru_result = all_results_summary[0]['Best_AUPRC']
            bilstm_result = all_results_summary[1]['Best_AUPRC']
            
            print(f"  GRU-4ch:    {gru_result:.4f}")
            print(f"  BiLSTM-4ch: {bilstm_result:.4f}")
            print(f"  Difference: {abs(gru_result - bilstm_result):.4f}")
            
            if bilstm_result < 0.60:
                print("\nNote: BiLSTM still performing poorly (AUPRC < 0.60)")
                print("      This confirms bidirectional architecture limitation on imbalanced data")
            
            print("\nRemaining work:")
            print("  Run 3: CNN-BiLSTM-4ch (~1.5 days)")
            print("  Run 4: CNN-BiLSTM-5ch (~1.5 days)")
            print("  Total remaining: ~3 days")
            
            cont = input("\nContinue with CNN models? (yes/no): ")
            if cont.lower() != 'yes':
                print("\n" + "="*80)
                print("STOPPED AFTER 2 RUNS")
                print("="*80)
                print("\nCompleted runs:")
                print("  - GRU-4ch optimized")
                print("  - BiLSTM-4ch validated")
                print("\nYou can proceed with thesis using:")
                print("  - GRU as primary model")
                print("  - BiLSTM for architectural comparison")
                print("  - Existing CNN-BiLSTM results (no GA)")
                break
    
    if run_idx == 4:
        print("\n" + "="*80)
        print("ALL 4 GA RUNS COMPLETE")
        print("="*80)
        
        df_summary = pd.DataFrame(all_results_summary)
        df_summary.to_csv('results/true_ga/summary_all_4runs.csv', index=False)
        
        print("\nFinal Summary:")
        for r in all_results_summary:
            print(f"  {r['Model']:20s}: AUPRC {r['Best_AUPRC']:.4f}")
        
        print("\nComparison with benchmarks:")
        gru_best = all_results_summary[0]['Best_AUPRC']
        print(f"  GRU-4ch (yours):       {gru_best:.4f}")
        print(f"  Rahman LSTM (2024):    0.7208")
        print(f"  CnnCrispr (2020):      0.679")
        print(f"  Improvement over Rahman: {((gru_best - 0.7208)/0.7208 * 100):.1f}%")
        
        if len(all_results_summary) >= 4:
            cnn_4ch = all_results_summary[2]['Best_AUPRC']
            cnn_5ch = all_results_summary[3]['Best_AUPRC']
            improvement = ((cnn_5ch - cnn_4ch) / cnn_4ch * 100)
            print(f"\n5-channel improvement for CNN: {improvement:.1f}%")
            print(f"  CNN-BiLSTM-4ch: {cnn_4ch:.4f}")
            print(f"  CNN-BiLSTM-5ch: {cnn_5ch:.4f}")
        
        print(f"\nCompleted at: {datetime.now()}")
        print("\nNext steps:")
        print("  1. Review results in results/true_ga/")
        print("  2. Compare with previous work")
        print("  3. Generate final figures")
        print("  4. Write thesis (Dec 30 - Jan 7)")
        print("  5. Submit Jan 9")
        print("\nAll results saved.")
        