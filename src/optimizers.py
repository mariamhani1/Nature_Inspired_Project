import random
import numpy as np
from src.train_utils import evaluate_hyperparameters

def random_hyperparameters():
    """Generate random hyperparameters within valid ranges"""
    # Ensure num_heads divides embedding_dim evenly
    embedding_dims = [64, 128, 256, 512]
    num_heads_options = {64: [2, 4], 128: [2, 4, 8], 256: [4, 8, 16], 512: [4, 8, 16]}

    embedding_dim = random.choice(embedding_dims)
    num_heads = random.choice(num_heads_options[embedding_dim])

    return {
        'embedding_dim': embedding_dim,
        'num_heads': num_heads,
        'num_layers': random.choice([1, 2, 3, 4]),
        'dim_feedforward': random.choice([256, 512, 1024, 2048]),
        'dropout': round(random.uniform(0.1, 0.5), 2),
        'learning_rate': random.choice([0.0001, 0.0005, 0.001, 0.005]),
        'batch_size': random.choice([32, 64, 128]),
        'epochs': 5  # Fixed for optimization
    }

# Particle Swarm Optimization (PSO)
class ParticleSwarmOptimization:
    def __init__(self, n_particles=5, n_iterations=5, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter

    def optimize(self, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
        print("\n" + "="*60)
        print("PARTICLE SWARM OPTIMIZATION")
        print("="*60)

        # Initialize particles (hyperparameter sets)
        particles = [random_hyperparameters() for _ in range(self.n_particles)]
        velocities = [{} for _ in range(self.n_particles)]

        # Evaluate initial particles
        fitness_scores = []
        for i, particle in enumerate(particles):
            print(f"\nEvaluating Particle {i+1}/{self.n_particles}")
            fitness, _ = evaluate_hyperparameters(
                particle, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
            )
            fitness_scores.append(fitness)
            print(f"Particle {i+1} Fitness: {fitness:.2f}%")

        # Personal best
        p_best = particles.copy()
        p_best_fitness = fitness_scores.copy()

        # Global best
        g_best_idx = np.argmax(p_best_fitness)
        g_best = p_best[g_best_idx].copy()
        g_best_fitness = p_best_fitness[g_best_idx]

        print(f"\nInitial Best Fitness: {g_best_fitness:.2f}%")

        # PSO iterations
        for iteration in range(self.n_iterations):
            print(f"\n--- PSO Iteration {iteration+1}/{self.n_iterations} ---")

            for i in range(self.n_particles):
                # Update velocity and position (simplified for discrete hyperparameters)
                # With some probability, move towards personal or global best
                if random.random() < 0.5:
                    # Move towards personal best
                    for key in p_best[i].keys():
                        if random.random() < self.c1 / (self.c1 + self.c2):
                            particles[i][key] = p_best[i][key]
                else:
                    # Move towards global best
                    for key in g_best.keys():
                        if random.random() < self.c2 / (self.c1 + self.c2):
                            particles[i][key] = g_best[key]

                # Add some randomness (exploration)
                if random.random() < 0.2:
                    particles[i] = random_hyperparameters()

                # Evaluate new position
                fitness, _ = evaluate_hyperparameters(
                    particles[i], X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
                )

                # Update personal best
                if fitness > p_best_fitness[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fitness[i] = fitness
                    print(f"Particle {i+1}: New Personal Best = {fitness:.2f}%")

                # Update global best
                if fitness > g_best_fitness:
                    g_best = particles[i].copy()
                    g_best_fitness = fitness
                    print(f"*** New Global Best = {g_best_fitness:.2f}% ***")

        print(f"\nPSO Final Best Fitness: {g_best_fitness:.2f}%")
        print("Best Hyperparameters:")
        for key, value in g_best.items():
            print(f"  {key}: {value}")

        return g_best, g_best_fitness

# Simulated Annealing
class SimulatedAnnealing:
    def __init__(self, initial_temp=100, cooling_rate=0.85, n_iterations=10):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.n_iterations = n_iterations

    def optimize(self, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
        print("\n" + "="*60)
        print("SIMULATED ANNEALING")
        print("="*60)

        # Initialize with random solution
        current_solution = random_hyperparameters()
        current_fitness, _ = evaluate_hyperparameters(
            current_solution, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
        )

        best_solution = current_solution.copy()
        best_fitness = current_fitness

        print(f"Initial Fitness: {current_fitness:.2f}%")

        temperature = self.initial_temp

        for iteration in range(self.n_iterations):
            print(f"\n--- SA Iteration {iteration+1}/{self.n_iterations} ---")
            print(f"Temperature: {temperature:.2f}")

            # Generate neighbor solution (perturb current solution)
            neighbor_solution = current_solution.copy()

            # Randomly modify 1-2 hyperparameters
            keys_to_modify = random.sample(list(neighbor_solution.keys()),
                                          random.randint(1, 2))

            temp_params = random_hyperparameters()
            for key in keys_to_modify:
                if key != 'epochs':
                    neighbor_solution[key] = temp_params[key]

            # Ensure num_heads divides embedding_dim
            if neighbor_solution['embedding_dim'] % neighbor_solution['num_heads'] != 0:
                neighbor_solution = random_hyperparameters()

            # Evaluate neighbor
            neighbor_fitness, _ = evaluate_hyperparameters(
                neighbor_solution, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
            )

            print(f"Current Fitness: {current_fitness:.2f}%")
            print(f"Neighbor Fitness: {neighbor_fitness:.2f}%")

            # Acceptance criterion
            delta = neighbor_fitness - current_fitness

            if delta > 0:
                # Accept better solution
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                print("✓ Accepted (Better solution)")

                if current_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
                    print(f"*** New Best Fitness = {best_fitness:.2f}% ***")
            else:
                # Accept worse solution with probability
                acceptance_prob = np.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
                    print(f"✓ Accepted (Probability: {acceptance_prob:.4f})")
                else:
                    print("✗ Rejected")

            # Cool down
            temperature *= self.cooling_rate

        print(f"\nSA Final Best Fitness: {best_fitness:.2f}%")
        print("Best Hyperparameters:")
        for key, value in best_solution.items():
            print(f"  {key}: {value}")

        return best_solution, best_fitness

# Ant Colony Optimization (ACO)
class AntColonyOptimization:
    def __init__(self, n_ants=5, n_iterations=5, evaporation_rate=0.5, alpha=1.0, beta=2.0):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance

        # Define discrete choices for each hyperparameter
        self.param_choices = {
            'embedding_dim': [64, 128, 256, 512],
            'num_layers': [1, 2, 3, 4],
            'dim_feedforward': [256, 512, 1024, 2048],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'batch_size': [32, 64, 128]
        }

        # Head choices depend on embedding dim
        self.head_choices = {
            64: [2, 4],
            128: [2, 4, 8],
            256: [4, 8, 16],
            512: [4, 8, 16]
        }

        # Initialize pheromone trails
        self.pheromones = {}
        for param, choices in self.param_choices.items():
            self.pheromones[param] = {choice: 1.0 for choice in map(str, choices)}

    def select_parameter(self, param):
        """Select parameter value based on pheromone trails"""
        choices = self.param_choices[param]
        pheromones = [self.pheromones[param][str(choice)] ** self.alpha for choice in choices]

        # Normalize to probabilities
        total = sum(pheromones)
        probabilities = [p / total for p in pheromones]

        # Select based on probability
        selected = np.random.choice(len(choices), p=probabilities)
        return choices[selected]

    def optimize(self, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
        print("\n" + "="*60)
        print("ANT COLONY OPTIMIZATION")
        print("="*60)

        best_solution = None
        best_fitness = 0

        for iteration in range(self.n_iterations):
            print(f"\n--- ACO Iteration {iteration+1}/{self.n_iterations} ---")

            # Each ant constructs a solution
            solutions = []
            fitness_scores = []

            for ant in range(self.n_ants):
                # Construct solution
                embedding_dim = self.select_parameter('embedding_dim')
                num_heads = random.choice(self.head_choices[embedding_dim])

                solution = {
                    'embedding_dim': embedding_dim,
                    'num_heads': num_heads,
                    'num_layers': self.select_parameter('num_layers'),
                    'dim_feedforward': self.select_parameter('dim_feedforward'),
                    'dropout': self.select_parameter('dropout'),
                    'learning_rate': self.select_parameter('learning_rate'),
                    'batch_size': self.select_parameter('batch_size'),
                    'epochs': 5
                }

                print(f"\nAnt {ant+1}/{self.n_ants}")

                # Evaluate solution
                fitness, _ = evaluate_hyperparameters(
                    solution, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
                )

                print(f"Ant {ant+1} Fitness: {fitness:.2f}%")

                solutions.append(solution)
                fitness_scores.append(fitness)

                # Update best solution
                if fitness > best_fitness:
                    best_solution = solution.copy()
                    best_fitness = fitness
                    print(f"*** New Best Fitness = {best_fitness:.2f}% ***")

            # Evaporate pheromones
            for param in self.pheromones:
                for choice in self.pheromones[param]:
                    self.pheromones[param][choice] *= (1 - self.evaporation_rate)

            # Deposit pheromones based on solution quality
            for solution, fitness in zip(solutions, fitness_scores):
                pheromone_deposit = fitness / 100.0  # Normalize

                for param, value in solution.items():
                    if param != 'epochs' and param != 'num_heads' and param in self.pheromones:
                        self.pheromones[param][str(value)] += pheromone_deposit

        print(f"\nACO Final Best Fitness: {best_fitness:.2f}%")
        print("Best Hyperparameters:")
        for key, value in best_solution.items():
            print(f"  {key}: {value}")

        return best_solution, best_fitness

# Tabu Search
class TabuSearch:
    def __init__(self, tabu_tenure=3, n_iterations=10, neighborhood_size=5):
        self.tabu_tenure = tabu_tenure
        self.n_iterations = n_iterations
        self.neighborhood_size = neighborhood_size

    def generate_neighborhood(self, solution):
        """Generate neighboring solutions"""
        neighbors = []

        for _ in range(self.neighborhood_size):
            neighbor = solution.copy()

            # Randomly modify 1 hyperparameter
            param_to_modify = random.choice([
                'embedding_dim', 'num_heads', 'num_layers', 'dim_feedforward',
                'dropout', 'learning_rate', 'batch_size'
            ])

            temp_params = random_hyperparameters()

            if param_to_modify == 'embedding_dim':
                # If changing embedding_dim, also update num_heads to be compatible
                neighbor['embedding_dim'] = temp_params['embedding_dim']
                neighbor['num_heads'] = temp_params['num_heads']
            else:
                neighbor[param_to_modify] = temp_params[param_to_modify]

            # Ensure compatibility
            if neighbor['embedding_dim'] % neighbor['num_heads'] != 0:
                neighbor = random_hyperparameters()

            neighbors.append(neighbor)

        return neighbors

    def solution_hash(self, solution):
        """Create a hashable representation of solution"""
        return str(sorted(solution.items()))

    def optimize(self, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
        print("\n" + "="*60)
        print("TABU SEARCH")
        print("="*60)

        # Initialize
        current_solution = random_hyperparameters()
        current_fitness, _ = evaluate_hyperparameters(
            current_solution, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
        )

        best_solution = current_solution.copy()
        best_fitness = current_fitness

        tabu_list = []

        print(f"Initial Fitness: {current_fitness:.2f}%")

        for iteration in range(self.n_iterations):
            print(f"\n--- Tabu Search Iteration {iteration+1}/{self.n_iterations} ---")

            # Generate neighborhood
            neighbors = self.generate_neighborhood(current_solution)

            # Evaluate neighbors
            best_neighbor = None
            best_neighbor_fitness = -float('inf')

            for i, neighbor in enumerate(neighbors):
                neighbor_hash = self.solution_hash(neighbor)

                # Skip if in tabu list
                if neighbor_hash in tabu_list:
                    continue

                print(f"Evaluating Neighbor {i+1}/{len(neighbors)}")

                fitness, _ = evaluate_hyperparameters(
                    neighbor, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
                )

                print(f"Neighbor {i+1} Fitness: {fitness:.2f}%")

                if fitness > best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = fitness

            # If we found a valid neighbor
            if best_neighbor is not None:
                current_solution = best_neighbor
                current_fitness = best_neighbor_fitness

                # Add to tabu list
                tabu_list.append(self.solution_hash(current_solution))

                # Maintain tabu list size
                if len(tabu_list) > self.tabu_tenure:
                    tabu_list.pop(0)

                print(f"Current Fitness: {current_fitness:.2f}%")

                # Update best solution
                if current_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
                    print(f"*** New Best Fitness = {best_fitness:.2f}% ***")
            else:
                print("No valid neighbors found (all tabu)")

        print(f"\nTabu Search Final Best Fitness: {best_fitness:.2f}%")
        print("Best Hyperparameters:")
        for key, value in best_solution.items():
            print(f"  {key}: {value}")

        return best_solution, best_fitness

# Grey Wolf Optimizer
class GreyWolfOptimizer:
    def __init__(self, n_wolves=5, n_iterations=5):
        self.n_wolves = n_wolves
        self.n_iterations = n_iterations

    def optimize(self, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
        print("\n" + "="*60)
        print("GREY WOLF OPTIMIZER")
        print("="*60)

        # Initialize wolf population
        wolves = [random_hyperparameters() for _ in range(self.n_wolves)]

        # Evaluate initial population
        fitness_scores = []
        for i, wolf in enumerate(wolves):
            print(f"\nEvaluating Wolf {i+1}/{self.n_wolves}")
            fitness, _ = evaluate_hyperparameters(
                wolf, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
            )
            fitness_scores.append(fitness)
            print(f"Wolf {i+1} Fitness: {fitness:.2f}%")

        # Identify alpha, beta, and delta wolves
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        alpha_idx, beta_idx, delta_idx = sorted_indices[0], sorted_indices[1], sorted_indices[2]

        alpha_pos = wolves[alpha_idx].copy()
        beta_pos = wolves[beta_idx].copy()
        delta_pos = wolves[delta_idx].copy()

        alpha_score = fitness_scores[alpha_idx]
        beta_score = fitness_scores[beta_idx]
        delta_score = fitness_scores[delta_idx]

        print(f"\nInitial Hierarchy:")
        print(f"  Alpha (best): {alpha_score:.2f}%")
        print(f"  Beta (2nd): {beta_score:.2f}%")
        print(f"  Delta (3rd): {delta_score:.2f}%")

        # Define hyperparameter choices for position updates
        param_choices = {
            'embedding_dim': [64, 128, 256, 512],
            'num_layers': [1, 2, 3, 4],
            'dim_feedforward': [256, 512, 1024, 2048],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'batch_size': [32, 64, 128]
        }

        # GWO iterations
        for iteration in range(self.n_iterations):
            a = 2 * (1 - iteration / self.n_iterations)  # Linearly decreasing from 2 to 0
            print(f"\n--- GWO Iteration {iteration+1}/{self.n_iterations} (a={a:.3f}) ---")

            for i in range(self.n_wolves):
                # Update each wolf's position based on alpha, beta, delta
                new_wolf = wolves[i].copy()

                # Randomly select which hyperparameters to update (mimicking position update)
                for param in param_choices.keys():
                    if param == 'embedding_dim':
                        continue  # Skip for now, handle with num_heads

                    r = np.random.random()

                    if r < 0.33:  # Influenced by alpha
                        new_wolf[param] = alpha_pos[param]
                    elif r < 0.66:  # Influenced by beta
                        new_wolf[param] = beta_pos[param]
                    else:  # Influenced by delta
                        new_wolf[param] = delta_pos[param]

                    # Add exploration with probability based on 'a'
                    if np.random.random() < a / 2:
                        new_wolf[param] = random.choice(param_choices[param])

                # Handle embedding_dim and num_heads together
                embedding_dims = [64, 128, 256, 512]
                num_heads_options = {64: [2, 4], 128: [2, 4, 8], 256: [4, 8, 16], 512: [4, 8, 16]}

                r = np.random.random()
                if r < 0.33:
                    new_wolf['embedding_dim'] = alpha_pos['embedding_dim']
                    new_wolf['num_heads'] = alpha_pos['num_heads']
                elif r < 0.66:
                    new_wolf['embedding_dim'] = beta_pos['embedding_dim']
                    new_wolf['num_heads'] = beta_pos['num_heads']
                else:
                    new_wolf['embedding_dim'] = delta_pos['embedding_dim']
                    new_wolf['num_heads'] = delta_pos['num_heads']

                # Exploration
                if np.random.random() < a / 2:
                    new_emb = random.choice(embedding_dims)
                    new_wolf['embedding_dim'] = new_emb
                    new_wolf['num_heads'] = random.choice(num_heads_options[new_emb])

                # Evaluate new position
                fitness, _ = evaluate_hyperparameters(
                    new_wolf, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
                )

                wolves[i] = new_wolf
                fitness_scores[i] = fitness

                # Update alpha, beta, delta
                if fitness > alpha_score:
                    delta_pos, delta_score = beta_pos.copy(), beta_score
                    beta_pos, beta_score = alpha_pos.copy(), alpha_score
                    alpha_pos, alpha_score = new_wolf.copy(), fitness
                    print(f"Wolf {i+1}: New Alpha = {fitness:.2f}%")
                elif fitness > beta_score:
                    delta_pos, delta_score = beta_pos.copy(), beta_score
                    beta_pos, beta_score = new_wolf.copy(), fitness
                    print(f"Wolf {i+1}: New Beta = {fitness:.2f}%")
                elif fitness > delta_score:
                    delta_pos, delta_score = new_wolf.copy(), fitness
                    print(f"Wolf {i+1}: New Delta = {fitness:.2f}%")

        print(f"\nGWO Final Best Fitness: {alpha_score:.2f}%")
        print("Best Hyperparameters:")
        for key, value in alpha_pos.items():
            print(f"  {key}: {value}")

        return alpha_pos, alpha_score

# Whale Optimization Algorithm
class WhaleOptimizationAlgorithm:
    def __init__(self, n_whales=5, n_iterations=5):
        self.n_whales = n_whales
        self.n_iterations = n_iterations

    def optimize(self, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
        print("\n" + "="*60)
        print("WHALE OPTIMIZATION ALGORITHM")
        print("="*60)

        # Initialize whale population
        whales = [random_hyperparameters() for _ in range(self.n_whales)]

        # Evaluate initial population
        fitness_scores = []
        for i, whale in enumerate(whales):
            print(f"\nEvaluating Whale {i+1}/{self.n_whales}")
            fitness, _ = evaluate_hyperparameters(
                whale, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
            )
            fitness_scores.append(fitness)
            print(f"Whale {i+1} Fitness: {fitness:.2f}%")

        # Find best whale (leader)
        best_idx = np.argmax(fitness_scores)
        best_whale = whales[best_idx].copy()
        best_score = fitness_scores[best_idx]

        print(f"\nInitial Best Fitness: {best_score:.2f}%")

        # Define hyperparameter choices
        param_choices = {
            'embedding_dim': [64, 128, 256, 512],
            'num_layers': [1, 2, 3, 4],
            'dim_feedforward': [256, 512, 1024, 2048],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'batch_size': [32, 64, 128]
        }

        embedding_dims = [64, 128, 256, 512]
        num_heads_options = {64: [2, 4], 128: [2, 4, 8], 256: [4, 8, 16], 512: [4, 8, 16]}

        # WOA iterations
        for iteration in range(self.n_iterations):
            a = 2 * (1 - iteration / self.n_iterations)  # Linearly decreasing from 2 to 0
            a2 = -1 + iteration * (-1 / self.n_iterations)  # Linearly decreasing from -1 to -2

            print(f"\n--- WOA Iteration {iteration+1}/{self.n_iterations} (a={a:.3f}) ---")

            for i in range(self.n_whales):
                r = np.random.random()
                A = 2 * a * r - a
                C = 2 * r

                p = np.random.random()

                new_whale = whales[i].copy()

                if p < 0.5:  # Encircling prey or searching for prey
                    if abs(A) < 1:  # Encircling prey
                        # Move towards best whale
                        for param in param_choices.keys():
                            if np.random.random() < abs(C) / 2:
                                new_whale[param] = best_whale[param]
                            else:
                                # Random walk
                                new_whale[param] = random.choice(param_choices[param])

                        # Handle embedding_dim and num_heads
                        if np.random.random() < abs(C) / 2:
                            new_whale['embedding_dim'] = best_whale['embedding_dim']
                            new_whale['num_heads'] = best_whale['num_heads']
                        else:
                            new_emb = random.choice(embedding_dims)
                            new_whale['embedding_dim'] = new_emb
                            new_whale['num_heads'] = random.choice(num_heads_options[new_emb])

                    else:  # Search for prey (exploration)
                        # Select random whale
                        random_whale = whales[np.random.randint(0, self.n_whales)]

                        for param in param_choices.keys():
                            if np.random.random() < 0.5:
                                new_whale[param] = random_whale[param]
                            else:
                                new_whale[param] = random.choice(param_choices[param])

                        # Handle embedding_dim and num_heads
                        if np.random.random() < 0.5:
                            new_whale['embedding_dim'] = random_whale['embedding_dim']
                            new_whale['num_heads'] = random_whale['num_heads']
                        else:
                            new_emb = random.choice(embedding_dims)
                            new_whale['embedding_dim'] = new_emb
                            new_whale['num_heads'] = random.choice(num_heads_options[new_emb])

                else:  # Spiral updating position
                    # Move in spiral towards best whale
                    for param in param_choices.keys():
                        if np.random.random() < 0.7:  # High probability to follow best
                            new_whale[param] = best_whale[param]
                        else:
                            new_whale[param] = random.choice(param_choices[param])

                    # Handle embedding_dim and num_heads
                    if np.random.random() < 0.7:
                        new_whale['embedding_dim'] = best_whale['embedding_dim']
                        new_whale['num_heads'] = best_whale['num_heads']
                    else:
                        new_emb = random.choice(embedding_dims)
                        new_whale['embedding_dim'] = new_emb
                        new_whale['num_heads'] = random.choice(num_heads_options[new_emb])

                # Evaluate new position
                fitness, _ = evaluate_hyperparameters(
                    new_whale, X_train, y_train, X_val, y_val, vocab_size, max_length, epochs=5, device=device
                )

                whales[i] = new_whale
                fitness_scores[i] = fitness

                print(f"Whale {i+1} Fitness: {fitness:.2f}%")

                # Update best whale
                if fitness > best_score:
                    best_whale = new_whale.copy()
                    best_score = fitness
                    print(f"*** New Best Fitness = {best_score:.2f}% ***")

        print(f"\nWOA Final Best Fitness: {best_score:.2f}%")
        print("Best Hyperparameters:")
        for key, value in best_whale.items():
            print(f"  {key}: {value}")

        return best_whale, best_score

def evaluate_pso_with_params(w, c1, c2, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
    """Evaluate PSO with specific parameters"""
    print(f"  Testing PSO: w={w:.2f}, c1={c1:.2f}, c2={c2:.2f}")
    pso = ParticleSwarmOptimization(n_particles=3, n_iterations=2, w=w, c1=c1, c2=c2)
    _, fitness = pso.optimize(X_train, y_train, X_val, y_val, vocab_size, max_length, device=device)
    return fitness

class HillClimbingPSOOptimizer:
    def __init__(self, max_iters=4):
        self.max_iters = max_iters

    def optimize(self, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
        print("\n" + "="*70)
        print("HILL CLIMBING: OPTIMIZING PSO PARAMETERS (w, c1, c2)")
        print("="*70)

        # Start with default
        w, c1, c2 = 0.7, 1.5, 1.5
        current_fitness = evaluate_pso_with_params(w, c1, c2, X_train, y_train, X_val, y_val, vocab_size, max_length, device=device)

        best_w, best_c1, best_c2 = w, c1, c2
        best_fitness = current_fitness

        print(f"Initial: w={w}, c1={c1}, c2={c2}, fitness={current_fitness:.2f}%")

        for it in range(self.max_iters):
            print(f"\n--- Iteration {it+1}/{self.max_iters} ---")

            improved = False
            # Try neighbors
            for dw in [-0.1, 0.1]:
                new_w = max(0.3, min(0.9, w + dw))
                fitness = evaluate_pso_with_params(new_w, c1, c2, X_train, y_train, X_val, y_val, vocab_size, max_length, device=device)
                if fitness > current_fitness:
                    w, current_fitness, improved = new_w, fitness, True
                    print(f"  ✓ Improved with w={w:.2f}")
                    break

            if not improved:
                for dc1 in [-0.3, 0.3]:
                    new_c1 = max(0.5, min(2.5, c1 + dc1))
                    fitness = evaluate_pso_with_params(w, new_c1, c2, X_train, y_train, X_val, y_val, vocab_size, max_length, device=device)
                    if fitness > current_fitness:
                        c1, current_fitness, improved = new_c1, fitness, True
                        print(f"  ✓ Improved with c1={c1:.2f}")
                        break

            if not improved:
                for dc2 in [-0.3, 0.3]:
                    new_c2 = max(0.5, min(2.5, c2 + dc2))
                    fitness = evaluate_pso_with_params(w, c1, new_c2, X_train, y_train, X_val, y_val, vocab_size, max_length, device=device)
                    if fitness > current_fitness:
                        c2, current_fitness, improved = new_c2, fitness, True
                        print(f"  ✓ Improved with c2={c2:.2f}")
                        break

            if not improved:
                print("  No improvement, stopping")
                break

            if current_fitness > best_fitness:
                best_w, best_c1, best_c2, best_fitness = w, c1, c2, current_fitness

        print(f"\n{'='*70}")
        print(f"OPTIMIZED PSO PARAMS: w={best_w:.2f}, c1={best_c1:.2f}, c2={best_c2:.2f}")
        print(f"Best Performance: {best_fitness:.2f}%")
        print(f"{'='*70}")

        return best_w, best_c1, best_c2, best_fitness

class FireflySAOptimizer:
    def __init__(self, n_fireflies=4, n_iters=3):
        self.n_fireflies = n_fireflies
        self.n_iters = n_iters

    def evaluate_sa(self, temp, cooling, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
        print(f"  Testing SA: temp={temp:.1f}, cooling={cooling:.3f}")
        sa = SimulatedAnnealing(initial_temp=temp, cooling_rate=cooling, n_iterations=4)
        _, fitness = sa.optimize(X_train, y_train, X_val, y_val, vocab_size, max_length, device=device)
        return fitness

    def optimize(self, X_train, y_train, X_val, y_val, vocab_size, max_length, device='cpu'):
        print("\n" + "="*70)
        print("FIREFLY: OPTIMIZING SA PARAMETERS (temp, cooling_rate)")
        print("="*70)

        # Initialize fireflies
        temps = [random.uniform(50, 150) for _ in range(self.n_fireflies)]
        coolings = [random.uniform(0.75, 0.95) for _ in range(self.n_fireflies)]
        fitness = [self.evaluate_sa(temps[i], coolings[i], X_train, y_train, X_val, y_val, vocab_size, max_length, device=device)
                   for i in range(self.n_fireflies)]

        best_idx = np.argmax(fitness)
        best_temp, best_cooling, best_fitness = temps[best_idx], coolings[best_idx], fitness[best_idx]

        print(f"Initial best: temp={best_temp:.1f}, cooling={best_cooling:.3f}, fitness={best_fitness:.2f}%")

        for it in range(self.n_iters):
            print(f"\n--- Iteration {it+1}/{self.n_iters} ---")

            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness[j] > fitness[i]:
                        # Move i towards j
                        if random.random() < 0.7:
                            temps[i] = (temps[i] + temps[j]) / 2
                            coolings[i] = (coolings[i] + coolings[j]) / 2

                        new_fitness = self.evaluate_sa(temps[i], coolings[i], X_train, y_train, X_val, y_val, vocab_size, max_length, device=device)

                        if new_fitness > fitness[i]:
                            fitness[i] = new_fitness
                            if new_fitness > best_fitness:
                                best_temp, best_cooling, best_fitness = temps[i], coolings[i], new_fitness
                                print(f"  ✓ New best: {best_fitness:.2f}%")

        print(f"\n{'='*70}")
        print(f"OPTIMIZED SA PARAMS: temp={best_temp:.1f}, cooling={best_cooling:.3f}")
        print(f"Best Performance: {best_fitness:.2f}%")
        print(f"{'='*70}")

        return best_temp, best_cooling, best_fitness
