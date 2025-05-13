import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load datasets
clean_data = np.loadtxt('data/source/clean_dataset.txt')
noisy_data = np.loadtxt('data/source/noisy_dataset.txt')

clean_X = clean_data[:, :-1]
clean_y = clean_data[:, -1]

noisy_X = noisy_data[:, :-1]
noisy_y = noisy_data[:, -1]

unique_labels = np.unique(noisy_y)

# Initialize and fit t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
concat_data = np.concatenate((clean_X, noisy_X), axis=0)
data_tsne = tsne.fit_transform(concat_data)

clean_num = np.shape(clean_y)[0]
noisy_num = np.shape(noisy_y)[0]

concat_labels = np.concatenate((clean_y, noisy_y), axis=0)

# Plotting
plt.figure(figsize=(8, 6))
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Visualization of Clean and Noisy Datasets with Labels")

def color(l):
    if l == 1:
        return (68/255, 1/255, 84/255)
    elif l == 2:
        return (49/255,104/255,142/255)
    elif l == 3:
        return (53/255,183/255,121/255)
    elif l == 4:
        return (253/255,231/255,37/255)
    
for label in unique_labels:
    label_indices = clean_y == label
    clean_data_tsne = data_tsne[:2000]
    plt.scatter(
        clean_data_tsne[label_indices, 0], 
        clean_data_tsne[label_indices, 1], 
        label=f"Class {int(label)} in Clean Set", 
        s=50, alpha=0.7, c=[color(label)]*np.shape(clean_data_tsne[label_indices, 0])[0], cmap='viridis', marker="*"
    )

for label in unique_labels:
    label_indices = noisy_y == label
    noisy_data_tsne = data_tsne[2000:]
    plt.scatter(
        noisy_data_tsne[label_indices, 0], 
        noisy_data_tsne[label_indices, 1], 
        label=f"Class {int(label)} in Noisy Set", 
        s=20, alpha=0.7, c=[color(label)]*np.shape(clean_data_tsne[label_indices, 0])[0], cmap='viridis', marker="s"
    )

plt.legend(title="Labels")
plt.show()
