"""
Part C - Direction 1: Optimization
Compare SGD vs SGD+MultiStepLR vs MomentGD+MultiStepLR on MLP baseline.
"""
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
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

results = {}

# Experiment 1: SGD only (no scheduler)
print("=" * 50)
print("Experiment 1: SGD (lr=0.06, no scheduler)")
print("=" * 50)
np.random.seed(42)
model1 = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
opt1 = nn.optimizer.SGD(init_lr=0.06, model=model1)
loss_fn1 = nn.op.MultiCrossEntropyLoss(model=model1, max_classes=10)
runner1 = nn.runner.RunnerM(model1, opt1, nn.metric.accuracy, loss_fn1)
start = time.time()
runner1.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=500, save_dir=r'./best_models_exp1', eval_iters=100)
results['SGD'] = {'time': time.time() - start, 'score': runner1.best_score, 'runner': runner1}
print(f"Time: {results['SGD']['time']:.1f}s, Best: {runner1.best_score:.4f}\n")

# Experiment 2: SGD + MultiStepLR
print("=" * 50)
print("Experiment 2: SGD (lr=0.06) + MultiStepLR")
print("=" * 50)
np.random.seed(42)
model2 = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
opt2 = nn.optimizer.SGD(init_lr=0.06, model=model2)
sch2 = nn.lr_scheduler.MultiStepLR(optimizer=opt2, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn2 = nn.op.MultiCrossEntropyLoss(model=model2, max_classes=10)
runner2 = nn.runner.RunnerM(model2, opt2, nn.metric.accuracy, loss_fn2, scheduler=sch2)
start = time.time()
runner2.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=500, save_dir=r'./best_models_exp2', eval_iters=100)
results['SGD+MultiStepLR'] = {'time': time.time() - start, 'score': runner2.best_score, 'runner': runner2}
print(f"Time: {results['SGD+MultiStepLR']['time']:.1f}s, Best: {runner2.best_score:.4f}\n")

# Experiment 3: MomentGD + MultiStepLR
print("=" * 50)
print("Experiment 3: MomentGD (lr=0.06, mu=0.9) + MultiStepLR")
print("=" * 50)
np.random.seed(42)
model3 = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
opt3 = nn.optimizer.MomentGD(init_lr=0.01, model=model3, mu=0.9)
sch3 = nn.lr_scheduler.MultiStepLR(optimizer=opt3, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn3 = nn.op.MultiCrossEntropyLoss(model=model3, max_classes=10)
runner3 = nn.runner.RunnerM(model3, opt3, nn.metric.accuracy, loss_fn3, scheduler=sch3)
start = time.time()
runner3.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=500, save_dir=r'./best_models_exp3', eval_iters=100)
results['MomentGD+MultiStepLR'] = {'time': time.time() - start, 'score': runner3.best_score, 'runner': runner3}
print(f"Time: {results['MomentGD+MultiStepLR']['time']:.1f}s, Best: {runner3.best_score:.4f}\n")

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
for name, res in results.items():
    print(f"  {name:30s} | Acc: {res['score']:.4f} | Time: {res['time']:.1f}s")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#E74C3C', '#3498DB', '#2ECC71']
names = list(results.keys())

for i, name in enumerate(names):
    r = results[name]['runner']
    axes[0].plot(r.dev_loss, color=colors[i], label=name, alpha=0.8)
    axes[1].plot(r.dev_scores, color=colors[i], label=name, alpha=0.8)

axes[0].set_xlabel('Evaluation step')
axes[0].set_ylabel('Dev Loss')
axes[0].legend()
axes[0].set_title('Validation Loss Comparison')

axes[1].set_xlabel('Evaluation step')
axes[1].set_ylabel('Dev Accuracy')
axes[1].legend()
axes[1].set_title('Validation Accuracy Comparison')

plt.tight_layout()
plt.savefig('./figs/optimization_comparison.png', dpi=150)
plt.show()
