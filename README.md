# Nature Inspired Project

This project implements a Nature-Inspired Computation pipeline using Deep Learning (Transformers) and various Metaheuristic algorithms for Hyperparameter Optimization and Explainable AI (XAI).

## Dataset
- **Name**: AG News (Text Classification)
- **Size**: ~20,000 Training Samples, 2,000 Validation Samples (High-dimensional NLP dataset).
- **Goal**: Classify news articles into 4 categories (World, Sports, Business, Sci/Tech).

## Project Pipeline & Algorithms

### Phase 1: Model Parameter Optimization (Step 1)
Tune hyperparameters (embedding size, layers, heads, etc.) of the Transformer model using 6 metaheuristics:
- **Particle Swarm Optimization (PSO)**
- **Simulated Annealing (SA)**
- **Ant Colony Optimization (ACO)**
- **Tabu Search (TS)**
- **Grey Wolf Optimizer (GWO)**
- **Whale Optimization Algorithm (WOA)**


### Phase 2: Parameter & Explainability Optimization

#### Step 3: Algorithm Parameter Tuning (Meta-Optimization)
Optimize the internal parameters of the optimizers themselves:
- **Hill Climbing (HC)**: Optimizes PSO parameters (`w`, `c1`, `c2`).
- **Firefly Algorithm (FA)**: Optimizes SA parameters (`temperature`, `cooling_rate`).

#### Step 4: Explainability Optimization (XAI)
Optimize **LIME** parameters (`num_features`, `num_samples`) to enhance the stability of explanations using 4 metaheuristics:
- **PSO**
- **SA**
- **Tabu Search**
- **ACO**

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the full pipeline:
   ```bash
   python main.py
   ```
