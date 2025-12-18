# Nature Inspired Project

This project implements a Nature-Inspired Computation pipeline using Deep Learning (Transformers) and various Metaheuristic algorithms for Hyperparameter Optimization and Explainable AI (XAI).

## Algorithms Used

### Phase 1: Hyperparameter Optimization
- Particle Swarm Optimization (PSO)
- Simulated Annealing (SA)
- Ant Colony Optimization (ACO)
- Tabu Search (TS)
- Grey Wolf Optimizer (GWO)
- Whale Optimization Algorithm (WOA)

### Phase 2: XAI Optimization
- Hill Climbing (HC) for PSO parameter tuning
- Firefly Algorithm (FA) for SA parameter tuning

### XAI Specific Optimizers
- PSO for LIME parameters
- SA for LIME parameters
- Tabu Search for LIME parameters
- ACO for LIME parameters

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the full pipeline:
   ```bash
   python main.py
   ```
