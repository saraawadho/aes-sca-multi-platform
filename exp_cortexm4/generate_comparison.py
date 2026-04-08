# Generate cross-dataset comparison summary figure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Side-Channel Analysis: Cross-Dataset Comparison Summary", fontsize=13, fontweight='bold')

# ── Left plot: CPA peak correlation by dataset ────────────────────────────────
datasets   = ['STM32F4\n(Cortex-M4)\nUnmasked', 'Cortex-M0\nUnmasked', 'AES-HD\n(FPGA)\nHD model', 'ASCAD\n(ATMega8515)\nMasked']
cpa_peaks  = [0.465, 0.40, 0.032, 0.028]
cpa_colors = ['steelblue', 'steelblue', 'darkorange', 'crimson']

ax = axes[0]
bars = ax.bar(datasets, cpa_peaks, color=cpa_colors, edgecolor='black', linewidth=0.7, width=0.5)
ax.axhline(0.05, color='gray', linestyle='--', linewidth=1, label='Weak signal threshold (~0.05)')
ax.set_ylabel("Peak |Pearson Correlation|", fontsize=10)
ax.set_title("CPA Performance (1st Order, HW/HD Model)", fontsize=10)
ax.set_ylim(0, 0.55)
ax.legend(fontsize=8)
for bar, val in zip(bars, cpa_peaks):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

outcomes = ['16/16 ✓', '16/16 ✓', '1/1 ✓\n(weak)', 'FAILED\n(masking)']
outcome_colors = ['green', 'green', 'darkorange', 'red']
for bar, txt, col in zip(bars, outcomes, outcome_colors):
    ax.text(bar.get_x() + bar.get_width()/2, 0.02, txt,
            ha='center', va='bottom', fontsize=7.5, color=col, fontweight='bold')

# ── Right plot: Attack success heatmap ───────────────────────────────────────
attack_methods = ['SPA', 'DPA\n(LSB)', 'CPA\n(HW/HD)', 'Template\nAttack', 'DL\n(MLP)']
ds_labels      = ['STM32F4 M4\n(own)', 'Cortex-M0\n(public)', 'AES-HD\n(FPGA)', 'ASCAD\n(masked)']

# 0=not tested, 1=failed, 2=partial, 3=success
results = np.array([
    [3, 3, 3, 3, 0],   # STM32F4
    [0, 2, 3, 0, 0],   # Cortex-M0
    [0, 3, 3, 0, 0],   # AES-HD
    [0, 1, 1, 0, 3],   # ASCAD
])

cmap_vals  = {0: '#e8e8e8', 1: '#e74c3c', 2: '#f39c12', 3: '#2ecc71'}
cell_labels = {0: 'N/A', 1: 'FAIL', 2: 'PARTIAL', 3: 'SUCCESS'}

ax2 = axes[1]
ax2.set_title("Attack Success Matrix", fontsize=10)
ax2.set_xlim(-0.5, len(attack_methods) - 0.5)
ax2.set_ylim(-0.5, len(ds_labels) - 0.5)

for i in range(len(ds_labels)):
    for j in range(len(attack_methods)):
        val = results[i, j]
        rect = mpatches.FancyBboxPatch((j - 0.45, i - 0.4), 0.9, 0.8,
                                        boxstyle="round,pad=0.05",
                                        facecolor=cmap_vals[val], edgecolor='white', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(j, i, cell_labels[val], ha='center', va='center',
                 fontsize=8, fontweight='bold',
                 color='white' if val in [1, 3] else ('black' if val == 0 else 'white'))

ax2.set_xticks(range(len(attack_methods)))
ax2.set_xticklabels(attack_methods, fontsize=9)
ax2.set_yticks(range(len(ds_labels)))
ax2.set_yticklabels(ds_labels, fontsize=9)
ax2.tick_params(length=0)
ax2.set_facecolor('#f5f5f5')

legend_patches = [
    mpatches.Patch(color='#2ecc71', label='SUCCESS'),
    mpatches.Patch(color='#f39c12', label='PARTIAL (5/16 bytes)'),
    mpatches.Patch(color='#e74c3c', label='FAILED (masking)'),
    mpatches.Patch(color='#e8e8e8', label='Not tested'),
]
ax2.legend(handles=legend_patches, loc='lower right', fontsize=7.5, framealpha=0.9)

plt.tight_layout()
plt.savefig("cross_dataset_comparison.png", dpi=200, bbox_inches='tight')
print("Saved: analysis/cross_dataset_comparison.png")
