# Project 1: Neural Network and CNN for MNIST Classification

**课程**: Neural Network and Deep Learning  
**姓名**: 陈展  
**学号**: [请填写你的学号]  
**代码**: https://github.com/CZcoco/FDU_DL.git  
**模型权重**: [请填写 ModelScope 链接]

---

## 1. MLP Baseline

### 1.1 模型架构

本项目使用一个两层 MLP 作为 baseline 模型：

- **输入层**: 784 维（28×28 像素展平）
- **隐藏层**: 600 个神经元，ReLU 激活函数
- **输出层**: 10 个神经元（对应 0-9 十个数字类别）

总参数量约为 784×600 + 600 + 600×10 + 10 = **476,410**。

权重初始化采用 Kaiming 初始化（`scale = sqrt(2/fan_in)`），偏置初始化为零。

### 1.2 训练设置

| 超参数 | 值 |
|--------|-----|
| 优化器 | SGD |
| 学习率 | 0.06 |
| 学习率调度 | MultiStepLR (milestones=[800, 2400, 4000], gamma=0.5) |
| Batch size | 32 |
| Epochs | 5 |
| Weight decay | 1e-4 |
| 训练集大小 | 50,000 |
| 验证集大小 | 10,000 |
| 数据归一化 | [0, 1] |

### 1.3 实验结果

MLP baseline 在验证集上达到 **95.1%** 的准确率，在测试集上达到 **95.2%** 的准确率。

### 1.4 学习曲线

![MLP Learning Curve](figs/Figure_1.png)

从学习曲线可以看出：
- 训练 loss 从约 2.3 快速下降到 0.2 以下
- 验证集准确率在前 1000 次迭代内快速上升到 90% 以上，之后缓慢提升
- 模型没有明显的过拟合现象

---

## 2. CNN Model and MLP-vs-CNN Comparison

### 2.1 CNN 架构

采用 LeNet-5 风格的 CNN 架构：

```
Conv(1→6, 5×5) → ReLU → MaxPool(2×2) → Conv(6→16, 5×5) → ReLU → MaxPool(2×2) → Flatten → Linear(256→120) → ReLU → Linear(120→84) → ReLU → Linear(84→10)
```

**维度变化**：
- 输入: [N, 1, 28, 28]
- Conv1 输出: [N, 6, 24, 24]
- MaxPool1 输出: [N, 6, 12, 12]
- Conv2 输出: [N, 16, 8, 8]
- MaxPool2 输出: [N, 16, 4, 4]
- Flatten: [N, 256]
- FC1: [N, 120]
- FC2: [N, 84]
- FC3: [N, 10]

总参数量约为 1×6×25 + 6 + 6×16×25 + 16 + 256×120 + 120 + 120×84 + 84 + 84×10 + 10 ≈ **42,000**。

### 2.2 conv2D 实现方法

卷积层采用 **im2col** 方法实现，将卷积运算转化为矩阵乘法：

1. **Forward**: 将输入的所有感受野 patch 展开为列向量，组成矩阵 `col [N, C_in×k×k, H_out×W_out]`，然后与 reshape 后的权重矩阵 `W [C_out, C_in×k×k]` 做矩阵乘法
2. **Backward**: 
   - `dW = sum_N(grad_reshaped @ col^T)`
   - `db = sum(grads, axis=(0,2,3))`
   - `dcol = W^T @ grad_reshaped`，然后通过 col2im 散布回输入形状

实现正确性通过有限差分法梯度检查验证，误差在 1e-11 量级。

### 2.3 训练设置

| 超参数 | 值 |
|--------|-----|
| 优化器 | SGD |
| 学习率 | 0.01 |
| 学习率调度 | MultiStepLR (milestones=[1500, 3000, 4500], gamma=0.5) |
| Batch size | 64 |
| Epochs | 3 |
| 训练集大小 | 50,000 |
| 验证集大小 | 10,000 |

注：CNN 学习率设为 0.01（小于 MLP 的 0.06），因为 CNN 的梯度信号更强，较大学习率容易导致训练不稳定。由于纯 NumPy 实现的卷积运算较慢，CNN 仅训练 3 epochs。

### 2.4 MLP vs CNN 对比

| 指标 | MLP | CNN |
|------|-----|-----|
| 验证集准确率 | 95.1% | **96.2%** |
| 测试集准确率 | 95.2% | — |
| 参数量 | ~476K | ~42K |
| 训练时间 (per epoch) | ~12s | ~180s |
| 训练 Epochs | 5 | 3 |

### 2.5 CNN 学习曲线

![CNN Learning Curve](figs/cnn_learning_curve.png)

### 2.6 讨论：为什么 CNN 更适合图像分类

1. **局部连接性**：CNN 通过卷积核只关注局部区域，能有效捕获图像的局部特征（边缘、纹理等），而 MLP 的全连接层将所有像素同等对待，忽略了空间结构
2. **权重共享**：同一个卷积核在整个图像上滑动，大幅减少参数量（42K vs 476K），降低过拟合风险
3. **平移不变性**：卷积操作天然具有平移不变性，同一特征无论出现在图像何处都能被检测到
4. **层次化特征提取**：多层卷积逐步提取从低级到高级的特征

实验结果验证了这些理论优势：CNN 用不到 1/10 的参数量，仅训练 3 epochs 就超过了 MLP 训练 5 epochs 的结果。

---

## 3. Additional Directions

### 3.1 Direction 1: Optimization

#### 3.1.1 实验设计

在 MLP baseline 上对比三种优化策略，每次只改变优化器/调度器，其他超参数保持一致：

| 实验 | 优化器 | 学习率 | 调度器 |
|------|--------|--------|--------|
| Exp 1 | SGD | 0.06 | 无 |
| Exp 2 | SGD | 0.06 | MultiStepLR (milestones=[800,2400,4000], γ=0.5) |
| Exp 3 | MomentGD (μ=0.9) | 0.01 | MultiStepLR (milestones=[800,2400,4000], γ=0.5) |

注：MomentGD 使用较小学习率 0.01，因为动量会放大梯度更新，较大学习率容易导致震荡。

#### 3.1.2 实验结果

| 方法 | 验证集准确率 | 训练时间 |
|------|-------------|----------|
| **SGD (无调度)** | **96.67%** | 58.2s |
| SGD + MultiStepLR | 94.98% | 55.4s |
| MomentGD + MultiStepLR | 95.94% | 75.5s |

#### 3.1.3 对比曲线

![Optimization Comparison](figs/optimization_comparison.png)

#### 3.1.4 分析与结论

1. **SGD 无调度器表现最好**：这说明当前 MultiStepLR 的 milestones 设置可能过早衰减了学习率。在训练后期模型仍有学习空间时，学习率被降低导致收敛变慢
2. **MomentGD 收敛更快但最终精度略低**：从学习曲线可以看出，MomentGD 在前期收敛速度明显快于 SGD，但由于使用了较小的学习率（0.01 vs 0.06），最终精度略低。如果适当调大学习率，MomentGD 有望获得更好结果
3. **学习率调度需要精心设计**：不恰当的调度策略反而会损害性能。milestones 的选择应该基于对训练曲线的观察，在 loss 趋于平稳时再降低学习率

### 3.2 Direction 5: Error Analysis and Visualization

#### 3.2.1 混淆矩阵分析

**MLP 混淆矩阵** (测试集准确率 95.2%):

![MLP Confusion Matrix](figs/confusion_matrix_mlp.png)

**CNN 混淆矩阵** (测试集):

![CNN Confusion Matrix](figs/confusion_matrix_cnn.png)

从 MLP 混淆矩阵可以观察到：
- 数字 1 的识别率最高（1115/1135 ≈ 98.2%）
- 数字 5 的识别率相对较低（831/892 ≈ 93.2%）
- 常见的混淆对包括：4→9（22例）、9→4（22例）、7→2（19例）、8→3（13例）

这些混淆是合理的，因为这些数字对在手写时形态相似。

#### 3.2.2 错误样本分析

![MLP Misclassified](figs/misclassified_mlp.png)

从最自信的错误分类样本中可以观察到：
- 模型容易将书写潦草或变形严重的数字误分类
- 形态相似的数字对（如 4/9、3/8、7/2）是主要的错误来源
- 部分样本即使人类也难以准确判断

#### 3.2.3 权重可视化

**MLP 第一层权重**:

![MLP Weights](figs/mlp_weights.png)

MLP 第一层的 600 个神经元中，每个神经元的权重可以 reshape 为 28×28 的图像。从可视化中可以看出：
- 不同神经元学习到了不同的空间模式
- 部分神经元对应特定数字的笔画特征
- 权重呈现出明显的空间结构，说明模型确实学习到了有意义的特征

**CNN 卷积核**:

![CNN Kernels](figs/cnn_kernels.png)

CNN 第一层的 6 个 5×5 卷积核展示了不同方向的边缘检测器，这与经典的图像处理理论一致。

---

## 4. Main Results Table

| 模型/方法 | 验证集准确率 | 测试集准确率 | 参数量 | 训练时间 |
|-----------|-------------|-------------|--------|----------|
| MLP (SGD + MultiStepLR) | 95.1% | 95.2% | ~476K | ~60s |
| CNN (SGD + MultiStepLR, 3 epochs) | 96.2% | — | ~42K | ~547s |
| MLP (SGD, 无调度) | 96.67% | — | ~476K | ~58s |
| MLP (SGD + MultiStepLR) | 94.98% | — | ~476K | ~55s |
| MLP (MomentGD + MultiStepLR) | 95.94% | — | ~476K | ~76s |

---

## 5. Detailed Visualization

本报告包含以下可视化内容：

1. **学习曲线**：MLP 和 CNN 的训练/验证 loss 及 accuracy 曲线（Section 1.4, 2.5）
2. **混淆矩阵**：MLP 和 CNN 在测试集上的 10×10 混淆矩阵（Section 3.2.1）
3. **错误样本可视化**：模型最自信的错误分类样本（Section 3.2.2）
4. **权重可视化**：MLP 第一层权重和 CNN 卷积核（Section 3.2.3）
5. **优化方法对比曲线**：三种优化策略的验证 loss/accuracy 对比（Section 3.1.3）

---

## 6. Discussion

### 6.1 为什么 CNN 比 MLP 更适合图像分类？

CNN 通过局部连接、权重共享和池化操作，天然地利用了图像数据的空间结构。相比之下，MLP 将图像展平为一维向量，完全丢失了像素间的空间关系。实验结果表明，CNN 用不到 1/10 的参数量就能超过 MLP 的性能。

### 6.2 CNN 是否提升了准确率？

是的。CNN 在仅训练 3 epochs 的情况下达到 96.2% 的验证准确率，超过了 MLP 训练 5 epochs 的 95.1%。如果给予更多训练时间，CNN 的优势会更加明显。

### 6.3 选择的两个方向及原因

- **Direction 1 (Optimization)**：优化方法是深度学习的核心组件，对比 SGD 和 Momentum 能直观展示动量对收敛速度的影响，同时学习率调度策略的效果也值得探究
- **Direction 5 (Error Analysis & Visualization)**：可视化分析能帮助理解模型的决策过程，混淆矩阵揭示了模型的系统性错误模式，权重可视化展示了模型学到的特征

### 6.4 最有信息量的分析

混淆矩阵分析最有信息量。它不仅给出了整体准确率，还揭示了模型在哪些类别对之间容易混淆（如 4/9、3/8），这些信息对于改进模型具有直接指导意义。

### 6.5 哪些样本对模型仍然困难？

从错误分析中发现，以下类型的样本最具挑战性：
- 书写极度潦草或变形的数字
- 形态相似的数字对（4 vs 9、3 vs 8、7 vs 2）
- 笔画不完整或有额外噪声的样本

这些困难样本反映了手写数字识别的固有挑战：同一数字的书写风格差异巨大，而不同数字之间可能存在形态重叠。
