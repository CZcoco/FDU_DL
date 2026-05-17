"""
Part C - Direction 5: Error Analysis and Visualization
- Confusion matrix
- Misclassified examples
- Weight visualization (MLP first layer + CNN kernels)
"""
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# Load test data
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs_norm = test_imgs / test_imgs.max()


def compute_confusion_matrix(model, imgs, labs, is_cnn=False):
    if is_cnn:
        imgs = imgs.reshape(-1, 1, 28, 28)
    logits = model(imgs)
    preds = np.argmax(logits, axis=1)
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(labs, preds):
        cm[true][pred] += 1
    return cm, preds, logits


def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    fig.colorbar(im)
    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_misclassified(imgs, labs, preds, logits, title, save_path, n=20):
    from mynn.op import softmax
    probs = softmax(logits)
    wrong_mask = preds != labs
    wrong_indices = np.where(wrong_mask)[0]

    confidences = probs[wrong_indices, preds[wrong_indices]]
    sorted_idx = np.argsort(-confidences)[:n]
    selected = wrong_indices[sorted_idx]

    fig, axes = plt.subplots(2, n // 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=14)
    for i, idx in enumerate(selected):
        ax = axes[i // (n // 2), i % (n // 2)]
        img = imgs[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'T:{labs[idx]} P:{preds[idx]}\n{confidences[sorted_idx[i]]:.2f}', fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_mlp_weights(model, save_path):
    W = model.layers[0].params['W']  # shape: [784, 600]
    fig, axes = plt.subplots(5, 10, figsize=(12, 6))
    fig.suptitle('MLP First Layer Weights (50 neurons)', fontsize=14)
    for i in range(50):
        ax = axes[i // 10, i % 10]
        weight = W[:, i].reshape(28, 28)
        ax.imshow(weight, cmap='RdBu_r', vmin=-weight.max(), vmax=weight.max())
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_cnn_kernels(model, save_path):
    conv1_W = model.layers[0].params['W']  # [6, 1, 5, 5]
    conv2_W = model.layers[3].params['W']  # [16, 6, 5, 5]

    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    fig.suptitle('CNN Convolution Kernels', fontsize=14)

    for i in range(6):
        ax = axes[0, i]
        kernel = conv1_W[i, 0]
        ax.imshow(kernel, cmap='RdBu_r', vmin=-kernel.max(), vmax=kernel.max())
        ax.set_title(f'Conv1-{i}', fontsize=8)
        ax.axis('off')
    for i in range(6, 8):
        axes[0, i].axis('off')

    for i in range(8):
        ax = axes[1, i]
        kernel = conv2_W[i, 0]
        ax.imshow(kernel, cmap='RdBu_r', vmin=-kernel.max(), vmax=kernel.max())
        ax.set_title(f'Conv2-{i}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == '__main__':
    # --- MLP Analysis ---
    print("Loading MLP model...")
    mlp_model = nn.models.Model_MLP()
    mlp_model.load_model(r'./best_models/best_model.pickle')

    print("Computing MLP confusion matrix...")
    cm_mlp, preds_mlp, logits_mlp = compute_confusion_matrix(mlp_model, test_imgs_norm, test_labs)
    acc_mlp = np.trace(cm_mlp) / cm_mlp.sum()
    print(f"MLP Test Accuracy: {acc_mlp:.4f}")
    plot_confusion_matrix(cm_mlp, f'MLP Confusion Matrix (Acc={acc_mlp:.4f})', './figs/confusion_matrix_mlp.png')

    print("Plotting MLP misclassified examples...")
    plot_misclassified(test_imgs_norm, test_labs, preds_mlp, logits_mlp,
                       'MLP: Most Confident Misclassifications', './figs/misclassified_mlp.png')

    print("Plotting MLP weights...")
    plot_mlp_weights(mlp_model, './figs/mlp_weights.png')

    # --- CNN Analysis ---
    print("\nLoading CNN model...")
    cnn_model = nn.models.Model_CNN()
    cnn_model.load_model(r'./best_models_cnn/best_model.pickle')

    print("Computing CNN confusion matrix...")
    cm_cnn, preds_cnn, logits_cnn = compute_confusion_matrix(cnn_model, test_imgs_norm, test_labs, is_cnn=True)
    acc_cnn = np.trace(cm_cnn) / cm_cnn.sum()
    print(f"CNN Test Accuracy: {acc_cnn:.4f}")
    plot_confusion_matrix(cm_cnn, f'CNN Confusion Matrix (Acc={acc_cnn:.4f})', './figs/confusion_matrix_cnn.png')

    print("Plotting CNN misclassified examples...")
    plot_misclassified(test_imgs_norm.reshape(-1, 1, 28, 28).reshape(-1, 784), test_labs, preds_cnn, logits_cnn,
                       'CNN: Most Confident Misclassifications', './figs/misclassified_cnn.png')

    print("Plotting CNN kernels...")
    plot_cnn_kernels(cnn_model, './figs/cnn_kernels.png')

    print("\nDone! All figures saved to ./figs/")
