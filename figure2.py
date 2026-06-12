import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
SAVE_DIR = os.path.join(base_dir, 'datasets', 'survival_plots')

# ── Load images ────────────────────────────────────────
img_a = mpimg.imread(os.path.join(SAVE_DIR, 'km_dfs_metabric.png'))
img_b = mpimg.imread(os.path.join(SAVE_DIR, 'km_os_metabric.png'))
img_c = mpimg.imread(os.path.join(SAVE_DIR, 'km_dfs_crosscohort.png'))
img_d = mpimg.imread(os.path.join(SAVE_DIR, 'km_os_crosscohort.png'))

# ── Create 2x2 figure ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

panels = [
    (axes[0, 0], img_a, 'a'),
    (axes[0, 1], img_b, 'b'),
    (axes[1, 0], img_c, 'c'),
    (axes[1, 1], img_d, 'd'),
]

for ax, img, label in panels:
    ax.imshow(img)
    ax.axis('off')
    ax.text(-0.01, 1.05, label, transform=ax.transAxes,
            fontsize=30, fontweight='bold', va='top', ha='right')

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.tight_layout(pad=1.5)

save_path = os.path.join(SAVE_DIR, 'figure_km_combined.png')
plt.savefig(save_path, dpi=450, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Combined KM figure saved → {save_path}")
