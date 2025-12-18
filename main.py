import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import pandas as pd
import numpy as np

from src.data_loader import load_data
from src.train_utils import train_model, evaluate_model, create_model_with_params
from src.optimizers import (
    ParticleSwarmOptimization, SimulatedAnnealing, AntColonyOptimization,
    TabuSearch, GreyWolfOptimizer, WhaleOptimizationAlgorithm,
    HillClimbingPSOOptimizer, FireflySAOptimizer
)
from src.xai import (
    LIMETextExplainerWrapper, PSO_LIME_Optimizer, SA_LIME_Optimizer,
    Tabu_LIME_Optimizer, ACO_LIME_Optimizer
)

def main():
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")

    # Load Data
    print("\n--- Loading Data ---")
    X_train, y_train, X_val, y_val, vocab_size, preprocessor = load_data()
    print(f"Vocab size: {vocab_size}")

    # Results dictionary
    results = {}
    
    # Baseline Model
    print("\n--- Training Baseline Model ---")
    baseline_params = {
        'embedding_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 10
    }
    
    model = create_model_with_params(baseline_params, vocab_size, preprocessor.max_length, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=baseline_params['learning_rate'])
    
    train_loader = DataLoader(
        TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train)),
        batch_size=baseline_params['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val)),
        batch_size=baseline_params['batch_size']
    )
    
    start_time = time.time()
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=baseline_params['epochs'], device=device)
    baseline_time = time.time() - start_time
    baseline_accuracy = evaluate_model(model, val_loader, device=device)
    
    results['Baseline'] = {
        'accuracy': baseline_accuracy,
        'time': baseline_time,
        'params': baseline_params
    }
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")

    # Phase 1: Hyperparameter Optimization
    print("\n--- Phase 1: Metaheuristic Hyperparameter Optimization ---")
    
    optimizers = {
        'Tabu Search': TabuSearch(tabu_tenure=3, n_iterations=8, neighborhood_size=4),
        'PSO': ParticleSwarmOptimization(n_particles=5, n_iterations=3),
        'Simulated Annealing': SimulatedAnnealing(initial_temp=100, cooling_rate=0.85, n_iterations=8),
        'ACO': AntColonyOptimization(n_ants=4, n_iterations=3),
        'Grey Wolf': GreyWolfOptimizer(n_wolves=5, n_iterations=3),
        'Whale': WhaleOptimizationAlgorithm(n_whales=5, n_iterations=3)
    }

    for name, optimizer in optimizers.items():
        print(f"\nRunning {name}...")
        start_time = time.time()
        best_params, fitness = optimizer.optimize(
            X_train, y_train, X_val, y_val, vocab_size, preprocessor.max_length, device=device
        )
        elapsed_time = time.time() - start_time
        
        results[name] = {
            'accuracy': fitness,
            'time': elapsed_time,
            'params': best_params
        }
        print(f"{name} Best Accuracy: {fitness:.2f}%")

    # Train final models with best parameters
    print("\n--- Training Final Models ---")
    final_results = {}
    
    for method, data in results.items():
        if method == 'Baseline':
            final_results[method] = {'accuracy': data['accuracy'], 'time': data['time']}
            continue

        print(f"Training {method} optimized model...")
        params = data['params'].copy()
        params['epochs'] = 10
        
        model = create_model_with_params(params, vocab_size, preprocessor.max_length, device=device)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        train_loader = DataLoader(
            TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train)),
            batch_size=params['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val)),
            batch_size=params['batch_size']
        )
        
        start_time = time.time()
        history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=params['epochs'], device=device)
        training_time = time.time() - start_time
        
        final_accuracy = history['val_acc'][-1]
        final_results[method] = {'accuracy': final_accuracy, 'time': training_time}
        print(f"{method} Final Accuracy: {final_accuracy:.2f}%")

    # Display Results
    print("\n--- Results Summary ---")
    comparison_df = pd.DataFrame({
        'Method': list(final_results.keys()),
        'Validation Accuracy (%)': [final_results[m]['accuracy'] for m in final_results.keys()],
        'Training Time (s)': [final_results[m]['time'] for m in final_results.keys()]
    })
    comparison_df = comparison_df.sort_values('Validation Accuracy (%)', ascending=False)
    print(comparison_df.to_string(index=False))

    # Visualization
    try:
        import matplotlib.pyplot as plt
        print("\n--- Generating Comparison Plot ---")
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Accuracy comparison
        methods = list(final_results.keys())
        accuracies = [final_results[m]['accuracy'] for m in methods]

        # Color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#9B59B6', '#1ABC9C']
        bars = axes[0].bar(methods, accuracies, color=colors[:len(methods)], alpha=0.8, edgecolor='black')

        axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Transformer Model Performance Comparison', fontsize=14, fontweight='bold')
        if accuracies:
            axes[0].set_ylim([min(accuracies) - 2, max(accuracies) + 2])
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0].tick_params(axis='x', rotation=25)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Training time comparison
        times = [final_results[m]['time'] for m in methods]
        bars2 = axes[1].bar(methods, times, color=colors[:len(methods)], alpha=0.8, edgecolor='black')

        axes[1].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        axes[1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1].tick_params(axis='x', rotation=25)

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s',
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig('transformer_metaheuristic_comparison.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'transformer_metaheuristic_comparison.png'")
    except ImportError:
        print("Matplotlib not installed or failed to plot. Skipping visualization.")


    # Phase 2: Parameter & Explainability Optimization
    print("\n--- Phase 2: XAI Optimization ---")
    
    # Select best model for XAI
    best_method = max(final_results.keys(), key=lambda k: final_results[k]['accuracy'])
    print(f"Using {best_method} model for XAI optimization")
    
    best_params = results[best_method]['params']
    xai_model = create_model_with_params(best_params, vocab_size, preprocessor.max_length, device=device)
    # Load model weights if we had saved them, or retrain briefly (retraining here for simplicity/consistency)
    optimizer = optim.Adam(xai_model.parameters(), lr=best_params['learning_rate'])
    train_loader = DataLoader(
        TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train)),
        batch_size=best_params['batch_size'], shuffle=True
    )
    # Train for fewer epochs for XAI usage
    train_model(xai_model, train_loader, val_loader, criterion, optimizer, epochs=5, verbose=False, device=device)

    # Hill Climbing & Firefly for Optimizer Parameters
    hc = HillClimbingPSOOptimizer(max_iters=3)
    opt_w, opt_c1, opt_c2, pso_perf = hc.optimize(
        X_train, y_train, X_val, y_val, vocab_size, preprocessor.max_length, device=device
    )

    ff = FireflySAOptimizer(n_fireflies=4, n_iters=2)
    opt_temp, opt_cool, sa_perf = ff.optimize(
        X_train, y_train, X_val, y_val, vocab_size, preprocessor.max_length, device=device
    )

    # LIME Parameter Optimization
    print("\nOPTIMIZING LIME PARAMETERS...")
   
    from datasets import load_dataset
    dataset = load_dataset('sh0416/ag_news', split='test[:10]')
    test_samples = [item['title'] + ' ' + item['description'] for item in dataset]
    
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    # XAI Optimizers
    xai_optimizers = {
        'PSO': PSO_LIME_Optimizer(n_particles=3, n_iters=2, class_names=class_names),
        'SA': SA_LIME_Optimizer(temp=20, cooling=0.85, n_iters=4, class_names=class_names),
        'Tabu': Tabu_LIME_Optimizer(tenure=2, n_iters=4, n_neighbors=3, class_names=class_names),
        'ACO': ACO_LIME_Optimizer(n_ants=3, n_iters=2, evap=0.5, class_names=class_names)
    }
    
    xai_results = {}
    for name, opt in xai_optimizers.items():
        print(f"\nRunning {name} for LIME...")
        params, score = opt.optimize(xai_model, preprocessor, test_samples, device=device)
        xai_results[name] = {'params': params, 'score': score}
        print(f"{name} Score: {score:.4f}")

    print("\n--- XAI Optimization Results ---")
    for name, data in xai_results.items():
        print(f"{name}: {data['score']:.4f}")

if __name__ == '__main__':
    main()
