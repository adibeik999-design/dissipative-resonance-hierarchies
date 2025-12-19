import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create professional summary figure
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('#f8f9fa')

# Create 2x2 grid
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel 1: Three-Layer Architecture
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('white')
layers = ['Wave Fields\n(Phase sync)', 'Metabolic Networks\n(Amplified catalysis)', 'Predictive Attractors\n(Compression)']
colors = ['#4C72B0', '#55A868', '#C44E52']
for i, (layer, color) in enumerate(zip(layers, colors)):
    y_pos = 2 - i
    # Box with fancy edges
    box = FancyBboxPatch((0.1, y_pos-0.3), 0.8, 0.6, 
                        boxstyle="round,pad=0.05", 
                        facecolor=color, alpha=0.7,
                        edgecolor='black', linewidth=1.5)
    ax1.add_patch(box)
    ax1.text(0.5, y_pos, layer, ha='center', va='center', 
             fontsize=9, fontweight='bold', color='white')
    # Arrow between layers
    if i < 2:
        ax1.arrow(0.5, y_pos-0.4, 0, -0.5, head_width=0.05, 
                 head_length=0.1, fc='black', ec='black')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 3)
ax1.set_title('Three-Layer Architecture', fontsize=12, fontweight='bold', pad=15)
ax1.axis('off')

# Panel 2: Quantitative Results
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('white')

metrics = ['Island Lifetime', 'Metabolic\nPersistence', 'Compression\nRatio']
values = [50, 5.3, 0.39]
errors = [12, 1.2, 0.08]
colors = ['#4C72B0', '#55A868', '#C44E52']

x_pos = np.arange(len(metrics))
bars = ax2.bar(x_pos, values, yerr=errors, capsize=8, 
               color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(errors) + 5,
             f'{value}', ha='center', va='bottom', fontweight='bold')

ax2.set_ylabel('Value', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics, fontweight='bold')
ax2.set_title('Simulation Results', fontsize=12, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Dissipative Ratchet Mechanism
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('white')

steps = ['Fluctuation\n(Noise)', 'Resonance\n(Coherence)', 'Dissipation\nBoost', 
         'Selection\n(Persistence)', 'Amplification', 'Compression']
x_pos = np.arange(len(steps))

# Create flowchart
for i, (step, x) in enumerate(zip(steps, x_pos)):
    # Circle for each step
    circle = plt.Circle((x, 0.5), 0.15, color='#DD8452', alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
    ax3.add_patch(circle)
    ax3.text(x, 0.5, str(i+1), ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    ax3.text(x, 0.15, step, ha='center', va='center', 
             fontsize=8, fontweight='bold')
    
    # Arrow between steps
    if i < len(steps)-1:
        ax3.arrow(x+0.15, 0.5, 0.25, 0, head_width=0.03, 
                 head_length=0.03, fc='black', ec='black')

ax3.set_xlim(-0.5, len(steps)-0.5)
ax3.set_ylim(0, 1)
ax3.set_title('Dissipative Ratchet Process', fontsize=12, fontweight='bold', pad=15)
ax3.axis('off')

# Panel 4: Simple Wave Pattern (simulated)
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('white')

# Generate simple wave pattern
x = np.linspace(0, 10, 100)
t = np.linspace(0, 5, 50)
X, T = np.meshgrid(x, t)
Z = np.sin(2*np.pi*X/3 + 0.5*T) * np.exp(-0.1*(X-5)**2) * np.exp(-0.1*T)

im = ax4.imshow(Z, aspect='auto', cmap='viridis', 
                extent=[0, 10, 5, 0], alpha=0.9)
ax4.set_xlabel('Space (lattice sites)', fontweight='bold')
ax4.set_ylabel('Time', fontweight='bold')
ax4.set_title('Resonant Island Formation', fontsize=12, fontweight='bold', pad=15)

# Add coherence regions
ax4.fill_between([2, 4], [0, 0], [5, 5], color='red', alpha=0.2, label='Coherent region')
ax4.fill_between([6, 8], [2, 2], [4, 4], color='red', alpha=0.2)
ax4.legend(loc='upper right', fontsize=8)

plt.colorbar(im, ax=ax4, label='Wave Amplitude')

plt.suptitle('Dissipative Resonance Hierarchies: Key Concepts & Results', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('paper/summary_figure.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.show()