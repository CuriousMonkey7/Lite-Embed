import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# 1. Define the 75 Classes (5 Categories) as per Appendix A.7
# ---------------------------------------------------------
classes = {
    "Cats": ["British Shorthair", "Maine Coon", "Bengal", "Abyssinian", "Persian", "Siamese", "Sphynx", "Ragdoll", "Scottish Fold", "Russian Blue", "Birman", "Bombay", "Burmese", "Savannah", "Norwegian Forest Cat"],
    "Dogs": ["Golden Retriever", "Chihuahua", "Labrador Retriever", "German Shepherd", "Bulldog", "Poodle", "Beagle", "Rottweiler", "Dachshund", "Boxer", "Siberian Husky", "Great Dane", "Doberman", "Pug", "Shih Tzu"],
    "Vehicles": ["SUV", "Pickup Truck", "Hatchback", "Sedan", "Convertible", "Minivan", "Sports Car", "Station Wagon", "Coupe", "Bus", "Motorcycle", "Scooter", "Tractor", "Ambulance", "Fire Engine"],
    "Food": ["Pizza", "Burger", "Sushi", "Pasta", "Taco", "Salad", "Steak", "Ice Cream", "Cake", "Pancakes", "Waffles", "Ramen", "Curry", "Sandwich", "Donut"],
    "Buildings": ["Skyscraper", "House", "Apartment", "Cabin", "Castle", "Mansion", "Barn", "Lighthouse", "Church", "Mosque", "Temple", "Hospital", "School", "Library", "Museum"],
}

all_classes = []
labels = []
for category, items in classes.items():
    all_classes.extend(items)
    labels.extend([category] * len(items))

# ---------------------------------------------------------
# 2. Extract Text Embeddings using OpenAI CLIP
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading official OpenAI CLIP Model (ViT-B/16) on {device}...")
model, preprocess = clip.load("ViT-B/16", device=device)

prompts = [f"a photo of a {cls.lower()}" for cls in all_classes]

print("Tokenizing and Extracting Text Embeddings...")
text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    # L2 normalize the embeddings
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

embeddings_np = text_features.cpu().float().numpy()

# ---------------------------------------------------------
# 3. Fit PCA
# ---------------------------------------------------------
print("Fitting PCA...")
pca = PCA()
projected_pca = pca.fit_transform(embeddings_np)
print(projected_pca.shape)
class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

def get_separation(class1, class2, pc_idx):
    idx1, idx2 = class_to_idx[class1], class_to_idx[class2]
    val1 = projected_pca[idx1, pc_idx]
    val2 = projected_pca[idx2, pc_idx]
    return abs(val1 - val2)

# ---------------------------------------------------------
# 4. Calculate Distances for Fig 5 Pairs
# ---------------------------------------------------------
cross_pairs = [("British Shorthair", "SUV"), ("British Shorthair", "Pickup Truck"), ("British Shorthair", "Hatchback")]
fine_pairs = [("Golden Retriever", "Chihuahua"), ("Maine Coon", "Bengal"), ("Maine Coon", "Abyssinian")]

# PC1 is index 0, PC4 is index 3
pc1_cross_dists = [get_separation(c1, c2, 0) for c1, c2 in cross_pairs]
pc1_fine_dists = [get_separation(c1, c2, 0) for c1, c2 in fine_pairs]

pc4_cross_dists = [get_separation(c1, c2, 3) for c1, c2 in cross_pairs]
pc4_fine_dists = [get_separation(c1, c2, 3) for c1, c2 in fine_pairs]

# ---------------------------------------------------------
# 5. t-SNE Projections (Coarse vs Fine Subspaces)
# ---------------------------------------------------------
print("Running t-SNE for Coarse and Fine spaces...")
dog_cat_indices = [i for i, label in enumerate(labels) if label in ["Dogs", "Cats"]]
dog_cat_labels = [labels[i] for i in dog_cat_indices]

# Top Right: Coarse Space (Original Embeddings)
dog_cat_embeddings =projected_pca[dog_cat_indices, :3] 
# Use init='random' to prevent PCA initialization crashes
tsne_coarse = TSNE(n_components=2, perplexity=10, metric='cosine', init='random', random_state=42)
tsne_results_coarse = tsne_coarse.fit_transform(dog_cat_embeddings)

# Bottom Right: Fine Space (Projected onto U_fine by dropping PC1)
# We take all components EXCEPT the first one (index 0)
dog_cat_pca_fine = projected_pca[dog_cat_indices, 3:] 
tsne_fine = TSNE(n_components=2, perplexity=10, metric='euclidean', init='random', random_state=42)
tsne_results_fine = tsne_fine.fit_transform(dog_cat_pca_fine)

# ---------------------------------------------------------
# 6. Plotting (Recreating Figure 5)
# ---------------------------------------------------------
print("Generating Plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.set_theme(style="whitegrid")

pair_labels = ["British Shorthair\nvs SUV", "British Shorthair\nvs Pickup Truck", "British Shorthair\nvs Hatchback", 
               "Golden Retriever\nvs Chihuahua", "Maine Coon\nvs Bengal", "Maine Coon\nvs Abyssinian"]

colors_pc1 = ['#F5A9BC']*3 + ['#A9D0F5']*3
colors_pc4 = ['#F5A9BC']*3 + ['#A9D0F5']*3

all_distances = pc1_cross_dists + pc1_fine_dists + pc4_cross_dists + pc4_fine_dists
max_y_limit = max(all_distances) * 1.15

# (a) Plot PC1 Bar Chart
bars_pc1 = axes[0, 0].bar(pair_labels, pc1_cross_dists + pc1_fine_dists, color=colors_pc1, edgecolor='black')
axes[0, 0].set_title("(a) PC1 Separation Analysis", fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel("Separation Distance")
axes[0, 0].set_ylim(0, max_y_limit)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].bar_label(bars_pc1, fmt='%.3f', padding=3)

# (b) Plot PC4 Bar Chart
bars_pc4 = axes[1, 0].bar(pair_labels, pc4_cross_dists + pc4_fine_dists, color=colors_pc4, edgecolor='black')
axes[1, 0].set_title("(b) PC4 Separation Analysis", fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel("Separation Distance")
axes[1, 0].set_ylim(0, max_y_limit)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].bar_label(bars_pc4, fmt='%.3f', padding=3)

# (c) Plot t-SNE Coarse (Top Right)
for label, marker, color in zip(["Dogs", "Cats"], ['o', '^'], ['#D87093', '#8FBC8F']):
    idx = [i for i, l in enumerate(dog_cat_labels) if l == label]
    axes[0, 1].scatter(tsne_results_coarse[idx, 0], tsne_results_coarse[idx, 1], label=label, marker=marker, c=color, s=80, edgecolors='gray')

# (d) Plot t-SNE Fine (Bottom Right)
for label, marker, color in zip(["Dogs", "Cats"], ['o', '^'], ['#D87093', '#8FBC8F']):
    idx = [i for i, l in enumerate(dog_cat_labels) if l == label]
    axes[1, 1].scatter(tsne_results_fine[idx, 0], tsne_results_fine[idx, 1], label=label, marker=marker, c=color, s=80, edgecolors='gray')

# Clean up axes and legends
axes[0, 1].legend()
axes[1, 1].legend()
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])

plt.tight_layout()
plt.savefig('clip_pca_analysis_fig5.png', dpi=300, bbox_inches='tight')
print("Figure saved successfully!")
plt.show()