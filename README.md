# Nature Inspired Project

This project implements a Nature-Inspired Computation pipeline using Deep Learning (Transformers) and various Metaheuristic algorithms for Hyperparameter Optimization and Explainable AI (XAI).

## Algorithms Used

### Deep Learning Model
- **Transformer Encoder**: Custom implementation with Multi-Head Attention and Feed-Forward Networks.
- **Positional Encoding**: Sinusoidal positional encodings to inject sequence order information.

### Phase 1: Hyperparameter Optimization (Metaheuristics)
- **Particle Swarm Optimization (PSO)**
- **Simulated Annealing (SA)**
- **Ant Colony Optimization (ACO)**
- **Tabu Search (TS)**
- **Grey Wolf Optimizer (GWO)**
- **Whale Optimization Algorithm (WOA)**

### Phase 2: Optimizer Parameter Tuning (Meta-Optimization)
- **Hill Climbing (HC)**: Used to optimize PSO parameters (`w`, `c1`, `c2`).
- **Firefly Algorithm (FA)**: Used to optimize SA parameters (`temperature`, `cooling_rate`).

### Phase 3: Explainable AI (XAI)
- **LIME (Local Interpretable Model-agnostic Explanations)**: For generating local explanations of model predictions.
- **XAI Optimizers**: The following algorithms are used to optimize LIME parameters (`num_features`, `num_samples`) for stability:
    - PSO
    - SA
    - Tabu Search
    - ACO

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the full pipeline:
   ```bash
   python main.py
   ```
