"""
Main simulation script for Dissipative Resonance Hierarchies model.
Implements the three-layer model: Wave Fields → Metabolism → Predictive Coding.
"""

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from tqdm import tqdm

class ResonantHierarchyModel:
    """Main class implementing the three-layer model."""
    
    def __init__(self, N=64, M=8, L=16, dt=0.05, seed=42):
        """Initialize model with parameters from the paper."""
        np.random.seed(seed)
        
        # Store dimensions
        self.N, self.M, self.L = N, M, L
        self.dt = dt
        
        # ===== LAYER 1: Wave Field Parameters =====
        self.gamma = 0.5      # Spatial coupling
        self.D = 0.2          # Noise intensity
        self.c = 0.1          # Nonlinear frequency shift
        
        # ===== LAYER 2: Metabolism Parameters =====
        self.eta = 1.0        # Wave-to-reaction coupling
        self.delta = 0.05     # Decay rate
        self.kappa0 = 0.1     # Baseline reaction rate
        
        # ===== LAYER 3: Predictive Coding Parameters =====
        self.eta_m = 0.8      # Metabolic-to-predictive coupling
        self.tau = 5.0        # Neural time constant
        
        # Initialize states
        self._initialize_states()
        self._initialize_connectivity()
    
    def _initialize_states(self):
        """Initialize all system states."""
        # Layer 1: Complex wave field
        self.z = 0.5 * (np.random.randn(self.N) + 1j*np.random.randn(self.N))
        self.omega = np.random.uniform(-0.1, 0.1, self.N)
        
        # Layer 2: Chemical concentrations (ensure positive)
        self.C = np.abs(np.random.randn(self.M)) * 0.1 + 0.05
        
        # Layer 3: Predictive attractor states
        self.x = np.zeros(self.L)
        
        # History buffers for analysis
        self.z_history = []
        self.C_history = []
        self.x_history = []
    
    def _initialize_connectivity(self):
        """Initialize connectivity matrices and mappings."""
        # Layer 1: Discrete Laplacian (periodic boundaries)
        main_diag = -2 * np.ones(self.N)
        off_diag = np.ones(self.N - 1)
        self.laplacian = self.gamma * diags(
            [off_diag, main_diag, off_diag],
            [-1, 0, 1],
            shape=(self.N, self.N)
        ).toarray()
        self.laplacian[0, -1] = self.gamma
        self.laplacian[-1, 0] = self.gamma
        
        # Layer 2: Simple autocatalytic network (Brusselator-like)
        self.nu = np.zeros((self.M, 3))  # Stoichiometry: M species × 3 reactions
        # Reaction 1: A → X
        self.nu[0, 0] = -1  # Consume A
        self.nu[2, 0] = 1   # Produce X
        # Reaction 2: 2X + Y → 3X
        self.nu[2, 1] = 1   # Net +1 X
        self.nu[3, 1] = -1  # Consume Y
        # Reaction 3: X → Y
        self.nu[2, 2] = -1  # Consume X
        self.nu[3, 2] = 1   # Produce Y
        
        # Reaction sites (which lattice sites influence each reaction)
        self.reaction_sites = [
            np.arange(self.N//4),           # Reaction 1 sites
            np.arange(self.N//2, 3*self.N//4), # Reaction 2 sites
            np.arange(3*self.N//4, self.N)   # Reaction 3 sites
        ]
        
        # Layer 3: Recurrent connectivity (sparse random)
        self.K = np.random.randn(self.L, self.L) * 0.3 / np.sqrt(self.L)
        self.K[np.abs(self.K) < 0.2] = 0  # Sparsify
        
        # Mapping from chemical species to Layer-3 units
        self.pi = np.random.randint(0, self.M, self.L)
    
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def step(self):
        """Advance the simulation by one time step."""
        # ===== LAYER 1: Stochastic CGLE =====
        noise = np.sqrt(self.D/self.dt) * (
            np.random.randn(self.N) + 1j*np.random.randn(self.N)
        )
        nonlinear = (1 + 1j*self.c) * np.abs(self.z)**2 * self.z
        diffusion = self.laplacian @ self.z
        
        self.z += self.dt * (
            (1 + 1j*self.omega) * self.z
            - nonlinear
            + diffusion
            + noise
        )
        
        # ===== LAYER 2: Wave-Amplified Metabolism =====
        A = np.abs(self.z)
        
        # Compute wave-boosted reaction rates
        kappa = np.zeros(3)
        for r in range(3):
            avg_A = np.mean(A[self.reaction_sites[r]])
            kappa[r] = self.kappa0 * (1 + self.eta * avg_A)
        
        # Compute reaction fluxes (mass-action)
        fluxes = np.zeros(3)
        # Reaction 1: A → X (rate = k1 * A)
        fluxes[0] = kappa[0] * self.C[0]
        # Reaction 2: 2X + Y → 3X (rate = k2 * X² * Y)
        fluxes[1] = kappa[1] * self.C[2]**2 * self.C[3]
        # Reaction 3: X → Y (rate = k3 * X)
        fluxes[2] = kappa[2] * self.C[2]
        
        # Update concentrations
        dC = self.nu @ fluxes - self.delta * self.C
        self.C += self.dt * dC
        self.C = np.maximum(self.C, 1e-8)  # Ensure non-negativity
        
        # ===== LAYER 3: Predictive Coding =====
        # Use time-averaged metabolic input (simple window)
        if len(self.C_history) >= 20:  # T_avg = 20
            C_avg = np.mean(self.C_history[-20:], axis=0)
            metabolic_input = self.eta_m * C_avg[self.pi]
            
            dx = (-self.x + self._sigmoid(self.K @ self.x + metabolic_input)) / self.tau
            self.x += self.dt * dx
        
        # Store history
        self.z_history.append(self.z.copy())
        self.C_history.append(self.C.copy())
        self.x_history.append(self.x.copy())
    
    def run(self, T=5000, show_progress=True):
        """Run simulation for T time steps."""
        iterator = tqdm(range(T)) if show_progress else range(T)
        for _ in iterator:
            self.step()
    
    def compute_compression_ratio(self, delta_t=5):
        """Compute predictive compression ratio (Equation 6 in paper)."""
        if len(self.C_history) < delta_t + 10:
            return 0.0
        
        # Use last 1000 time points for stable estimate
        n_points = min(1000, len(self.C_history) - delta_t)
        C_array = np.array(self.C_history[-n_points-delta_t:])
        x_array = np.array(self.x_history[-n_points-delta_t:])
        
        # Simple linear decoder from x to C
        # In a full implementation, you'd train this decoder
        decoder = np.linalg.lstsq(
            x_array[:-delta_t],
            C_array[delta_t:],
            rcond=None
        )[0]
        
        # Predict future C from current x
        C_pred = x_array[:-delta_t] @ decoder
        
        # Compute mean baseline error
        C_mean = np.mean(C_array[delta_t:], axis=0)
        baseline_error = np.mean(np.sum((C_array[delta_t:] - C_mean)**2, axis=1))
        
        # Compute prediction error
        pred_error = np.mean(np.sum((C_array[delta_t:] - C_pred)**2, axis=1))
        
        # Compression ratio
        CR = 1 - pred_error / baseline_error
        return max(CR, 0.0)  # Clip negative values
    
    def analyze_resonant_islands(self):
        """Analyze phase coherence to detect resonant islands."""
        if len(self.z_history) == 0:
            return []
        
        # Use last 100 time points
        z_array = np.array(self.z_history[-100:])
        phases = np.angle(z_array)
        
        # Compute local phase coherence
        coherence = np.zeros(self.N)
        for i in range(self.N):
            # Phase coherence with neighbors
            left = (i - 1) % self.N
            right = (i + 1) % self.N
            phase_diff = np.abs(
                np.sin((phases[:, i] - phases[:, left])/2)
            ).mean()
            coherence[i] = 1 - phase_diff
        
        # Identify islands (coherence > threshold)
        threshold = 0.7
        islands = []
        in_island = False
        start = 0
        
        for i in range(self.N):
            if coherence[i] > threshold and not in_island:
                in_island = True
                start = i
            elif coherence[i] <= threshold and in_island:
                in_island = False
                if i - start > 2:  # Minimum island size
                    islands.append((start, i))
        
        return islands, coherence

# ===== VISUALIZATION FUNCTIONS =====
def create_summary_figure(model, save_path="results/figures/summary.png"):
    """Create the three-panel summary figure for the paper."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Panel A: Wave amplitude spatiotemporal plot
    z_array = np.array(model.z_history[-500:])  # Last 500 time steps
    A = np.abs(z_array).T  # Transpose for correct orientation
    
    im = axes[0].imshow(A, aspect='auto', cmap='viridis',
                       extent=[0, 500*model.dt, 0, model.N])
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Lattice Site")
    axes[0].set_title("A. Wave Amplitude $A_i(t)$ (Resonant Islands)")
    plt.colorbar(im, ax=axes[0], label='Amplitude')
    
    # Panel B: Total metabolic concentration
    C_array = np.array(model.C_history)
    total_C = np.sum(C_array, axis=1)
    time = np.arange(len(total_C)) * model.dt
    
    axes[1].plot(time, total_C, 'b-', linewidth=1)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Total Metabolic Concentration")
    axes[1].set_title("B. Metabolic Activity $\sum_k C_k(t)$")
    axes[1].grid(True, alpha=0.3)
    
    # Panel C: Compression ratio over time (sliding window)
    window = 200
    step = 50
    CR_times = []
    CR_values = []
    
    for i in range(0, len(model.C_history) - window, step):
        # Create temporary model copy for window analysis
        sub_model = ResonantHierarchyModel(N=model.N, M=model.M, L=model.L)
        sub_model.C_history = model.C_history[i:i+window]
        sub_model.x_history = model.x_history[i:i+window]
        CR = sub_model.compute_compression_ratio()
        CR_times.append(time[i + window//2])
        CR_values.append(CR)
    
    axes[2].plot(CR_times, CR_values, 'r-', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Compression Ratio $\mathcal{CR}$")
    axes[2].set_title("C. Predictive Compression Efficiency")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-0.1, 0.5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("Running Dissipative Resonance Hierarchies simulation...")
    
    # Initialize and run model
    model = ResonantHierarchyModel()
    model.run(T=3000)  # 3000 time steps
    
    # Analyze results
    islands, coherence = model.analyze_resonant_islands()
    CR = model.compute_compression_ratio()
    
    print(f"\n=== Simulation Results ===")
    print(f"Resonant islands detected: {len(islands)}")
    for i, (start, end) in enumerate(islands):
        print(f"  Island {i+1}: sites {start}-{end} (size: {end-start})")
    print(f"Predictive Compression Ratio: {CR:.3f}")
    
    # Create summary figure for paper
    create_summary_figure(model, "paper/simulation_results_figure.png")
    print("\nFigure saved to 'paper/simulation_results_figure.png'")
    
    # Save quantitative metrics
    metrics = {
        'compression_ratio': float(CR),
        'n_islands': len(islands),
        'avg_island_size': np.mean([e-s for s,e in islands]) if islands else 0,
        'total_metabolism': float(np.sum(model.C_history[-1])),
        'avg_wave_coherence': float(np.mean(coherence))
    }
    
    import json
    with open('results/metrics/simulation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Metrics saved to 'results/metrics/simulation_metrics.json'")
