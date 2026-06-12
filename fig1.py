import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
SAVE_DIR = os.path.join(base_dir, 'datasets', 'figures')

# ── Load images ────────────────────────────────────────
img_a = mpimg.imread(os.path.join(SAVE_DIR, 'auc_expA_comparison.png'))
img_b = mpimg.imread(os.path.join(SAVE_DIR, 'auc_expB_comparison.png'))
img_c = mpimg.imread(os.path.join(SAVE_DIR, 'paired_sensitivity_age_meno.png'))

# ── Create figure ──────────────────────────────────────
# Row 1: two plots side by side
# Row 2: one wide plot spanning full width
fig = plt.figure(figsize=(18, 14))

# Panel a — top left
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(img_a)
ax1.axis('off')
ax1.text(-0.05, 1.05, 'a', transform=ax1.transAxes,
         fontsize=30, fontweight='bold', va='top', ha='right')

# Panel b — top right
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(img_b)
ax2.axis('off')
ax2.text(-0.05, 1.05, 'b', transform=ax2.transAxes,
         fontsize=30, fontweight='bold', va='top', ha='right')

# Panel c — full bottom row
ax3 = fig.add_subplot(2, 1, 2)
ax3.imshow(img_c)
ax3.axis('off')
ax3.text(-0.023, 1.05, 'c', transform=ax3.transAxes,
         fontsize=30, fontweight='bold', va='top', ha='right')

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.tight_layout(pad=1.5)

save_path = os.path.join(SAVE_DIR, 'figure_combined.png')
plt.savefig(save_path, dpi=450, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Combined figure saved → {save_path}")
