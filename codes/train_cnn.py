import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import time

np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# Reshape to [N, 1, 28, 28] for CNN
train_imgs = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

print(f"Train: {train_imgs.shape}, Valid: {valid_imgs.shape}")

cnn_model = nn.models.Model_CNN()
optimizer = nn.optimizer.SGD(init_lr=0.01, model=cnn_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1500, 3000, 4500], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=10)

runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64, scheduler=scheduler)

print("Starting CNN training...")
start = time.time()
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=3, log_iters=50, save_dir=r'./best_models_cnn', eval_iters=50)
elapsed = time.time() - start
print(f"Training time: {elapsed:.1f}s")
print(f"Best score: {runner.best_score:.4f}")

_, axes = plt.subplots(1, 2, figsize=(12, 4))
_.set_tight_layout(True)
plot(runner, axes)
plt.savefig('./figs/cnn_learning_curve.png', dpi=150)
plt.show()
