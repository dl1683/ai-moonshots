"""
Generate comprehensive multi-modality figure for CTI Universal Law paper.
Shows: (1) NLP 12-arch LOAO, (2) alpha bar chart by architecture family,
(3) Cross-modal comparison scatter, (4) K=100 regime analysis.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

# --- Load data ---
with open('results/cti_kappa_loao_per_dataset.json') as f:
    loao_data = json.load(f)

with open('results/cti_resnet50_cifar100.json') as f:
    cnn_data = json.load(f)

with open('results/cti_vit_cifar100.json') as f:
    vit_cifar100 = json.load(f)

with open('results/cti_noisefloor_analysis.json') as f:
    noisefloor = json.load(f)

# --- Figure setup ---
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

# ============ Panel 1: LOAO alpha bar chart (12 NLP architectures) ============
ax1 = fig.add_subplot(gs[0, 0])

loao_results = loao_data['loao_results']
arch_names = list(loao_results.keys())
alphas = [loao_results[a]['alpha'] for a in arch_names]
mean_alpha = loao_data['loao_alpha_mean']

# Color by family
FAMILY_COLORS = {
    'pythia': '#1f77b4',
    'gpt-neo': '#ff7f0e',
    'Qwen': '#2ca02c',
    'OLMo': '#d62728',
    'TinyLlama': '#9467bd',
    'Mistral': '#8c564b',
    'Falcon': '#e377c2',
    'rwkv': '#7f7f7f',
}

colors = []
for a in arch_names:
    for key, col in FAMILY_COLORS.items():
        if key.lower() in a.lower():
            colors.append(col)
            break
    else:
        colors.append('#17becf')

short_names = [a.replace('pythia-', 'Py-').replace('gpt-neo-', 'GPT-Neo-')
                .replace('TinyLlama-1.1B-intermediate-step-1431k-3T', 'TinyLlama')
                .replace('Mistral-7B-v0.3', 'Mistral-7B')
                .replace('rwkv-4-169m-pile', 'RWKV-4')
                .replace('Falcon-H1-0.5B-Base', 'Falcon-H1')
                .replace('OLMo-1B-hf', 'OLMo-1B') for a in arch_names]

bars = ax1.bar(range(len(arch_names)), alphas, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.axhline(mean_alpha, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_alpha:.3f}')
ax1.axhspan(mean_alpha - mean_alpha * 0.05, mean_alpha + mean_alpha * 0.05,
            alpha=0.15, color='red', label='5% band')
ax1.set_xticks(range(len(arch_names)))
ax1.set_xticklabels(short_names, rotation=55, ha='right', fontsize=7)
ax1.set_ylabel(r'LOAO $\hat{\alpha}$')
ax1.set_title(f'(A) NLP 12-arch LOAO\nCV={loao_data["loao_alpha_cv"]:.3f} (pre-reg threshold 0.25)', fontsize=10)
ax1.legend(loc='upper left', fontsize=7)
ax1.set_ylim(1.35, 1.60)

# ============ Panel 2: Architecture-family alpha comparison ============
ax2 = fig.add_subplot(gs[0, 1])

families = [
    ('NLP Decoders\n(12 arch)', [loao_results[a]['alpha'] for a in arch_names], '#1f77b4'),
    ('ViT K=10\n(2 arch)', [0.592, 0.630], '#2ca02c'),
    ('ViT K=100\n(1 arch)', [3.899], '#ff7f0e'),
    ('ResNet50\nK=100', [4.418], '#d62728'),
]

positions = []
x_pos = 0
box_data = []
x_labels = []
x_ticks = []

for fam_name, alpha_list, color in families:
    if len(alpha_list) > 1:
        bp = ax2.boxplot(alpha_list, positions=[x_pos], widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1),
                        flierprops=dict(markersize=4))
    else:
        ax2.scatter([x_pos], alpha_list, color=color, s=80, zorder=5, marker='D')
    x_ticks.append(x_pos)
    x_labels.append(fam_name)
    x_pos += 1.2

ax2.set_xticks(x_ticks)
ax2.set_xticklabels(x_labels, fontsize=8)
ax2.set_ylabel(r'Fitted $\alpha$')
ax2.set_title('(B) Alpha by Architecture Family\n(Form universal; constant varies by modality)', fontsize=10)
ax2.set_yscale('log')
ax2.set_ylim(0.5, 10)
ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

# ============ Panel 3: K=100 regime — ViT vs CNN comparison ============
ax3 = fig.add_subplot(gs[1, 0])

# Load ViT-Base CIFAR-10 data for comparison (use LOAO r as proxy)
K_values = [10, 10, 100, 100]
r_values = [0.901, 0.982, 0.773, 0.749]
labels_3 = ['ViT-Base\nCIFAR-10', 'ViT-Large\nCIFAR-10', 'ViT-Base\nCIFAR-100', 'ResNet50\nCIFAR-100']
colors_3 = ['#2ca02c', '#2ca02c', '#ff7f0e', '#d62728']
markers_3 = ['s', 's', 'o', '^']

for i, (k, r, lab, col, mark) in enumerate(zip(K_values, r_values, labels_3, colors_3, markers_3)):
    ax3.scatter(k, r, color=col, marker=mark, s=100, zorder=5, label=lab)

# Add noise floor line
k_sim_vals = noisefloor['simulations']
for sim in k_sim_vals:
    if sim['K'] == 10:
        ax3.axhline(sim['r_mean'], color='#2ca02c', linestyle=':', alpha=0.7, linewidth=1,
                    label=f'Sim E[r] K=10 ({sim["r_mean"]:.3f})')
        break
for sim in k_sim_vals:
    if 'CNN K=100' in sim['config'] and 'test-only' not in sim['config']:
        ax3.axhline(sim['r_mean'], color='orange', linestyle=':', alpha=0.7, linewidth=1,
                    label=f'Sim E[r] K=100 ({sim["r_mean"]:.3f})')
        break

ax3.set_xlabel('Number of classes K')
ax3.set_ylabel('Pearson r (kappa vs logit q)')
ax3.set_title('(C) K=100 Regime Analysis\nViT and CNN give same r at K=100', fontsize=10)
ax3.legend(loc='lower right', fontsize=7)
ax3.set_xscale('log')
ax3.set_xticks([10, 100])
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax3.set_ylim(0.6, 1.05)
ax3.grid(True, alpha=0.3)

# ============ Panel 4: Summary evidence table as text ============
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary_lines = [
    ("CTI Universal Law Evidence Summary", True),
    ("", False),
    ("logit(q_norm) = alpha*kappa - beta*log(K-1) + C0", False),
    ("", False),
    ("NLP (12 arch, LOAO)", True),
    ("  alpha = 1.477 +/- 0.034, CV = 0.019", False),
    ("  R^2 = 0.955 (per-dataset C0)", False),
    ("  Spans 2021-2025, 125M-7B params", False),
    ("  RWKV (pure RNN) PASS: alpha=2.887", False),
    ("", False),
    ("Vision Cross-Modal", True),
    ("  ViT-Large CIFAR-10 (K=10): R^2=0.964", False),
    ("  ViT+CNN CIFAR-100 (K=100): r~0.75", False),
    ("  Architecture-independent at same K", False),
    ("", False),
    ("Noise-floor (K=100 sim)", True),
    ("  E[r] = 0.875 under perfect law", False),
    ("  Obs r=0.75: class structure limits", False),
    ("", False),
    ("Causal Tests", True),
    ("  Frozen do-intervention: r=0.899", False),
    ("  Ortho factorial: Arm A r=0.899", False),
    ("  Negative control: r=0.000 PASS", False),
]

y_start = 0.97
for line, is_bold in summary_lines:
    ax4.text(0.02, y_start, line, transform=ax4.transAxes,
             fontsize=8.5,
             verticalalignment='top',
             fontweight='bold' if is_bold else 'normal',
             fontfamily='serif')
    y_start -= 0.045

ax4.set_title('(D) Evidence Summary', fontsize=10)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

plt.suptitle('CTI Universal Law: Functional Form Derived from EVT,\nValidated Across NLP (12 arch), ViT, and CNN',
             fontsize=12, fontweight='bold', y=1.01)

outpath = 'results/figures/fig_cti_multimodal_summary.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
