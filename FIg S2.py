import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
SAVE_DIR = os.path.join(base_dir, 'datasets', 'final_model')

img_a = mpimg.imread(os.path.join(SAVE_DIR, 'example_patient_shap.png'))
img_b = mpimg.imread(os.path.join(SAVE_DIR, 'example_patient_shap2.png'))

fig, axes = plt.subplots(2, 1, figsize=(14, 20))

panels = [
    (axes[0], img_a),
    (axes[1], img_b),
]

for ax, img in panels:
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(hspace=0.05)
plt.tight_layout()

# Panel labels — a top, b bottom
label_positions = [
    ('a', 0.02, 0.97),
    ('b', 0.02, 0.49),
]

for label, x, y in label_positions:
    fig.text(x, y, label, fontsize=32, fontweight='bold',
             va='top', ha='left', transform=fig.transFigure)

save_path = os.path.join(SAVE_DIR, 'shap_waterfall.png')
plt.savefig(save_path, dpi=450, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Combined figure saved -> {save_path}")
