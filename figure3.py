import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
SAVE_DIR = os.path.join(base_dir, 'datasets', 'shap_plots')

img_a = mpimg.imread(os.path.join(SAVE_DIR, 'shap_pie_clinical_genomic.png'))
img_b = mpimg.imread(os.path.join(SAVE_DIR, 'shap_bar_all_features.png'))
img_c = mpimg.imread(os.path.join(SAVE_DIR, 'shap_beeswarm.png'))
img_d = mpimg.imread(os.path.join(SAVE_DIR, 'ablation_auc_boxplot.png'))

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

panels = [
    (axes[0, 0], img_a),
    (axes[0, 1], img_b),
    (axes[1, 0], img_c),
    (axes[1, 1], img_d),
]

for ax, img in panels:
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.tight_layout()

# Add panel labels in figure coordinates — fixed positions
label_positions = [
    ('a', 0.02, 0.97),
    ('b', 0.51, 0.97),
    ('c', 0.02, 0.49),
    ('d', 0.51, 0.49),
]

for label, x, y in label_positions:
    fig.text(x, y, label, fontsize=32, fontweight='bold',
             va='top', ha='left', transform=fig.transFigure)

save_path = os.path.join(SAVE_DIR, 'figure_ablation_combined.png')
plt.savefig(save_path, dpi=450, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Combined figure saved -> {save_path}")
