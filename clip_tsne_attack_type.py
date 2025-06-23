import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from sklearn.manifold import TSNE
import pandas as pd
from dataloader_attack_type import FAS_Dataset_AttackTypes
import argparse

parser = argparse.ArgumentParser(description="config")
parser.add_argument("--test_dataset", type=str)
args = parser.parse_args()
test_dataset = args.test_dataset

# Configuration
CLIP_MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
ROOT_PATH = "/shared/shared"
PLOT_SAVE_PATH = "tsne_plots"
os.makedirs(PLOT_SAVE_PATH, exist_ok=True)

# Class label mapping
attack_type_labels = {
    0: 'real',
    1: 'print',
    2: 'replay',
    3: 'mask',
    4: 'partial',
    5: 'fakehead'
}

if test_dataset in ['O', 'C', 'I', 'M']:
    del attack_type_labels[3], attack_type_labels[4], attack_type_labels[5]  # Remove mask, partial, fakehead for OULU-NPU, CASIA-FASD, and Replay-Attack

# Load dataset
dataset = FAS_Dataset_AttackTypes(root=ROOT_PATH, protocol=[test_dataset], train=False, size=224)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Load CLIP model
clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE, jit=False)
clip_model.eval()

# Extract features
image_features = []
image_labels = []

print("Extracting CLIP image features...")
for images, labels in tqdm(dataloader):
    pil_images = [ToPILImage()(img.cpu()) for img in images]
    processed = torch.stack([clip_preprocess(pil) for pil in pil_images]).to(DEVICE)

    with torch.no_grad():
        feats = clip_model.encode_image(processed)
        feats /= feats.norm(dim=-1, keepdim=True)
        image_features.append(feats.cpu().numpy())
        image_labels.extend(labels.cpu().numpy())

image_features = np.concatenate(image_features, axis=0)
image_labels = np.array(image_labels)

# t-SNE
print("Running t-SNE...")
perplexity_val = min(30, len(image_features) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity_val, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(image_features)

# DataFrame for plotting
df = pd.DataFrame(tsne_results, columns=['tsne-1', 'tsne-2'])
df['label'] = image_labels
df['class'] = df['label'].map(attack_type_labels)

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="tsne-1", y="tsne-2",
    hue="class",
    palette="Set2",
    data=df,
    alpha=0.7,
    s=25
)

# Remove axes, grid, and title
plt.axis('off')
plt.grid(False)
# plt.legend(title="", bbox_to_anchor=(1.05, 1), loc='lower right')
plt.legend(title="", loc='lower right')
plt.tight_layout()

# Save
dataset_name = {
    'O': 'OULU-NPU',
    'C': 'CASIA-FASD',
    'I': 'Replay-Attack',
    'M': 'MSU-MFSD',
    'W': 'WMCA'
}

save_path = os.path.join(PLOT_SAVE_PATH, f"tsne_multi_classes_{dataset_name[test_dataset]}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"✅ Clean t-SNE plot saved to {save_path}")