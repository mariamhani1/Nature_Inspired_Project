import torch
import numpy as np
import random
from lime.lime_text import LimeTextExplainer

class LIMETextExplainerWrapper:
    def __init__(self, model, preprocessor, class_names, device='cpu'):
        self.model = model
        self.preprocessor = preprocessor
        self.class_names = class_names
        self.device = device
        self.model.eval()

    def predict_proba(self, texts):
        """Prediction function for LIME"""
        encoded = self.preprocessor.encode(texts)
        with torch.no_grad():
            inputs = torch.LongTensor(encoded).to(self.device)
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def explain(self, text, num_features=10, num_samples=500):
        """Generate LIME explanation"""
        explainer = LimeTextExplainer(class_names=self.class_names)

        exp = explainer.explain_instance(
            text,
            self.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )

        return exp

    def get_stability_score(self, text, num_runs=3):
        """Measure explanation stability"""
        explanations = []
        for _ in range(num_runs):
            exp = self.explain(text, num_features=10, num_samples=500)
            explanations.append(dict(exp.as_list()))

        # Check consistency of top features
        all_features = set()
        for exp in explanations:
            all_features.update(exp.keys())

        # Calculate variance of feature importances
        variances = []
        for feature in all_features:
            values = [exp.get(feature, 0) for exp in explanations]
            variances.append(np.var(values))

        stability = 1.0 / (1.0 + np.mean(variances))
        return stability

class PSO_LIME_Optimizer:
    def __init__(self, n_particles=3, n_iters=2, class_names=None):
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.class_names = class_names

    def evaluate(self, num_features, num_samples, model, preprocessor, test_texts, device='cpu'):
        explainer = LIMETextExplainerWrapper(model, preprocessor, self.class_names, device)
        scores = []
        for text in test_texts[:2]:  # Use 2 samples for speed
            score = explainer.get_stability_score(text, num_runs=2)
            scores.append(score)
        return np.mean(scores)

    def optimize(self, model, preprocessor, test_texts, device='cpu'):
        print("\n" + "="*60)
        print("PSO: OPTIMIZING LIME PARAMETERS")
        print("="*60)

        # Parameter choices
        feat_choices = [5, 10, 15, 20]
        samp_choices = [100, 300, 500, 1000]

        # Initialize particles
        particles = [{'num_features': random.choice(feat_choices),
                     'num_samples': random.choice(samp_choices)}
                    for _ in range(self.n_particles)]

        fitness = [self.evaluate(p['num_features'], p['num_samples'], model, preprocessor, test_texts, device)
                  for p in particles]

        p_best = particles.copy()
        p_best_fit = fitness.copy()

        g_best_idx = np.argmax(p_best_fit)
        g_best = p_best[g_best_idx].copy()
        g_best_fit = p_best_fit[g_best_idx]

        print(f"Initial best: {g_best}, score={g_best_fit:.4f}")

        for it in range(self.n_iters):
            print(f"\n--- Iteration {it+1}/{self.n_iters} ---")

            for i in range(self.n_particles):
                # Move towards global best
                if random.random() < 0.6:
                    particles[i] = g_best.copy()
                else:
                    particles[i] = {'num_features': random.choice(feat_choices),
                                   'num_samples': random.choice(samp_choices)}

                fit = self.evaluate(particles[i]['num_features'], particles[i]['num_samples'],
                                   model, preprocessor, test_texts, device)

                if fit > p_best_fit[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fit[i] = fit

                if fit > g_best_fit:
                    g_best = particles[i].copy()
                    g_best_fit = fit
                    print(f"  ✓ New best: {g_best}, score={g_best_fit:.4f}")

        print(f"\n{'='*60}")
        print(f"OPTIMIZED LIME: {g_best}, score={g_best_fit:.4f}")
        print(f"{'='*60}")

        return g_best, g_best_fit

class SA_LIME_Optimizer:
    def __init__(self, temp=30, cooling=0.85, n_iters=5, class_names=None):
        self.temp = temp
        self.cooling = cooling
        self.n_iters = n_iters
        self.class_names = class_names

    def evaluate(self, num_features, num_samples, model, preprocessor, test_texts, device='cpu'):
        explainer = LIMETextExplainerWrapper(model, preprocessor, self.class_names, device)
        scores = []
        for text in test_texts[:2]:
            score = explainer.get_stability_score(text, num_runs=2)
            scores.append(score)
        return np.mean(scores)

    def get_neighbor(self, current):
        feat_choices = [5, 10, 15, 20]
        samp_choices = [100, 300, 500, 1000]

        neighbor = current.copy()
        if random.random() < 0.5:
            neighbor['num_features'] = random.choice(feat_choices)
        else:
            neighbor['num_samples'] = random.choice(samp_choices)
        return neighbor

    def optimize(self, model, preprocessor, test_texts, device='cpu'):
        print("\n" + "="*60)
        print("SA: OPTIMIZING LIME PARAMETERS")
        print("="*60)

        current = {'num_features': 10, 'num_samples': 300}
        current_fit = self.evaluate(current['num_features'], current['num_samples'],
                                    model, preprocessor, test_texts, device)

        best = current.copy()
        best_fit = current_fit
        temp = self.temp

        print(f"Initial: {current}, score={current_fit:.4f}")

        for it in range(self.n_iters):
            print(f"\n--- Iteration {it+1}/{self.n_iters} (T={temp:.2f}) ---")

            neighbor = self.get_neighbor(current)
            neighbor_fit = self.evaluate(neighbor['num_features'], neighbor['num_samples'],
                                        model, preprocessor, test_texts, device)

            delta = neighbor_fit - current_fit

            if delta > 0 or random.random() < np.exp(delta / temp):
                current = neighbor
                current_fit = neighbor_fit
                print(f"  ✓ Accepted: {current}, score={current_fit:.4f}")

                if current_fit > best_fit:
                    best = current.copy()
                    best_fit = current_fit
                    print(f"  *** New best: {best_fit:.4f}")

            temp *= self.cooling

        print(f"\n{'='*60}")
        print(f"OPTIMIZED LIME: {best}, score={best_fit:.4f}")
        print(f"{'='*60}")

        return best, best_fit

class Tabu_LIME_Optimizer:
    def __init__(self, tenure=2, n_iters=5, n_neighbors=3, class_names=None):
        self.tenure = tenure
        self.n_iters = n_iters
        self.n_neighbors = n_neighbors
        self.class_names = class_names

    def evaluate(self, num_features, num_samples, model, preprocessor, test_texts, device='cpu'):
        explainer = LIMETextExplainerWrapper(model, preprocessor, self.class_names, device)
        scores = []
        for text in test_texts[:2]:
            score = explainer.get_stability_score(text, num_runs=2)
            scores.append(score)
        return np.mean(scores)

    def optimize(self, model, preprocessor, test_texts, device='cpu'):
        print("\n" + "="*60)
        print("TABU: OPTIMIZING LIME PARAMETERS")
        print("="*60)

        feat_choices = [5, 10, 15, 20]
        samp_choices = [100, 300, 500, 1000]

        current = {'num_features': 10, 'num_samples': 300}
        current_fit = self.evaluate(current['num_features'], current['num_samples'],
                                    model, preprocessor, test_texts, device)

        best = current.copy()
        best_fit = current_fit
        tabu = []

        print(f"Initial: {current}, score={current_fit:.4f}")

        for it in range(self.n_iters):
            print(f"\n--- Iteration {it+1}/{self.n_iters} ---")

            neighbors = [{'num_features': random.choice(feat_choices),
                         'num_samples': random.choice(samp_choices)}
                        for _ in range(self.n_neighbors)]

            best_neighbor = None
            best_neighbor_fit = -1

            for n in neighbors:
                n_hash = f"{n['num_features']}_{n['num_samples']}"
                if n_hash in tabu:
                    continue

                fit = self.evaluate(n['num_features'], n['num_samples'],
                                   model, preprocessor, test_texts, device)

                if fit > best_neighbor_fit:
                    best_neighbor = n
                    best_neighbor_fit = fit

            if best_neighbor:
                current = best_neighbor
                current_fit = best_neighbor_fit
                tabu.append(f"{current['num_features']}_{current['num_samples']}")

                if len(tabu) > self.tenure:
                    tabu.pop(0)

                if current_fit > best_fit:
                    best = current.copy()
                    best_fit = current_fit
                    print(f"  ✓ New best: {best}, score={best_fit:.4f}")

        print(f"\n{'='*60}")
        print(f"OPTIMIZED LIME: {best}, score={best_fit:.4f}")
        print(f"{'='*60}")

        return best, best_fit

class ACO_LIME_Optimizer:
    def __init__(self, n_ants=3, n_iters=3, evap=0.5, class_names=None):
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.evap = evap
        self.class_names = class_names

        self.feat_choices = [5, 10, 15, 20]
        self.samp_choices = [100, 300, 500, 1000]

        self.pheromones = {
            'feat': {str(c): 1.0 for c in self.feat_choices},
            'samp': {str(c): 1.0 for c in self.samp_choices}
        }

    def evaluate(self, num_features, num_samples, model, preprocessor, test_texts, device='cpu'):
        explainer = LIMETextExplainerWrapper(model, preprocessor, self.class_names, device)
        scores = []
        for text in test_texts[:2]:
            score = explainer.get_stability_score(text, num_runs=2)
            scores.append(score)
        return np.mean(scores)

    def select(self, param_name, choices):
        pheromones = [self.pheromones[param_name][str(c)] for c in choices]
        total = sum(pheromones)
        probs = [p/total for p in pheromones]
        return choices[np.random.choice(len(choices), p=probs)]

    def optimize(self, model, preprocessor, test_texts, device='cpu'):
        print("\n" + "="*60)
        print("ACO: OPTIMIZING LIME PARAMETERS")
        print("="*60)

        best = None
        best_fit = 0

        for it in range(self.n_iters):
            print(f"\n--- Iteration {it+1}/{self.n_iters} ---")

            solutions = []
            fitness = []

            for ant in range(self.n_ants):
                sol = {
                    'num_features': self.select('feat', self.feat_choices),
                    'num_samples': self.select('samp', self.samp_choices)
                }

                fit = self.evaluate(sol['num_features'], sol['num_samples'],
                                   model, preprocessor, test_texts, device)

                solutions.append(sol)
                fitness.append(fit)

                print(f"  Ant {ant+1}: {sol}, score={fit:.4f}")

                if fit > best_fit:
                    best = sol.copy()
                    best_fit = fit
                    print(f"    ✓ New best!")

            # Update pheromones
            for p in self.pheromones:
                for c in self.pheromones[p]:
                    self.pheromones[p][c] *= (1 - self.evap)

            for sol, fit in zip(solutions, fitness):
                deposit = fit
                self.pheromones['feat'][str(sol['num_features'])] += deposit
                self.pheromones['samp'][str(sol['num_samples'])] += deposit

        print(f"\n{'='*60}")
        print(f"OPTIMIZED LIME: {best}, score={best_fit:.4f}")
        print(f"{'='*60}")

        return best, best_fit
