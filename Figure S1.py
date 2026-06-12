import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
SAVE_DIR = os.path.join(base_dir, 'datasets', 'survival_plots')

# ── Load images ────────────────────────────────────────
# Row 1: DFS — Luminal A, Luminal B, Triple-Negative
# Row 2: OS  — Luminal A, Luminal B, Triple-Negative
img_a = mpimg.imread(os.path.join(SAVE_DIR, 'km_dfs_Luminal_A_cv.png'))
img_b = mpimg.imread(os.path.join(SAVE_DIR, 'km_dfs_Luminal_B_cv.png'))
img_c = mpimg.imread(os.path.join(SAVE_DIR, 'km_dfs_TripleNegative_cv.png'))
img_d = mpimg.imread(os.path.join(SAVE_DIR, 'km_os_Luminal_A_cv.png'))
img_e = mpimg.imread(os.path.join(SAVE_DIR, 'km_os_Luminal_B_cv.png'))
img_f = mpimg.imread(os.path.join(SAVE_DIR, 'km_os_TripleNegative_cv.png'))

# ── Create 2x3 figure ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(27, 14))

panels = [
    (axes[0, 0], img_a, 'a'),
    (axes[0, 1], img_b, 'b'),
    (axes[0, 2], img_c, 'c'),
    (axes[1, 0], img_d, 'd'),
    (axes[1, 1], img_e, 'e'),
    (axes[1, 2], img_f, 'f'),
]

for ax, img, label in panels:
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.tight_layout()

# ── Panel labels in figure coordinates ────────────────
label_positions = [
    ('a', 0.02,  0.98),
    ('b', 0.36,  0.98),
    ('c', 0.685, 0.98),
    ('d', 0.02,  0.51),
    ('e', 0.36,  0.51),
    ('f', 0.685, 0.51),
]

for label, x, y in label_positions:
    fig.text(x, y, label, fontsize=30, fontweight='bold',
             va='top', ha='left', transform=fig.transFigure)

save_path = os.path.join(SAVE_DIR, 'figure_km_subtypes.png')
plt.savefig(save_path, dpi=450, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Subtype KM figure saved → {save_path}")
