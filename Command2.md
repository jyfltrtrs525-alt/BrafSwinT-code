请对我的 BraFSwin 训练框架进行改造，实现一种 **Delayed Dynamic Feature-Space Oversampling (DDFO)** 机制。

目标：

当前版本（train_head.py）是在训练开始前一次性提取 frozen feature，然后一次性进行 feature-space oversampling。

我希望改为：

* backbone 与 classifier 联合训练；
* feature-space synthetic samples 动态生成；
* 只有当特征空间已经相对稳定后才启动 synthetic feature generation；
* synthetic samples 每隔若干 epoch 重新生成；
* 使用 Tomek Links 仅清理 synthetic samples；
* synthetic features 只用于增强分类边界学习，而不是替代真实图像训练。

---

## 总体训练流程

Stage 1：Representation Warmup

训练开始时：

* 使用真实图像与真实标签训练；
* 不进行任何 feature-space oversampling；
* 不生成 synthetic features；
* 不进行 Tomek Links cleaning；

目的：

让 backbone 学到稳定的 shared feature representation。

---

## Stage 2：Delayed Dynamic Oversampling

当满足以下条件时启动：

```python
val_4c_acc >= activation_threshold
```

例如：

```python
activation_threshold = 0.75
```

为了避免偶然波动：

要求连续 P 个 epoch 满足条件，例如：

```python
activation_patience = 3
```

只有连续满足后才激活动态 oversampling。

---

## 高置信样本筛选

激活后，

每次更新 synthetic pool 时：

首先提取当前训练集的 shared features。

对于每个样本：

```python
x
mal_label
mut_label
```

得到：

```python
p_mal
p_mut
```

计算：

```python
mal_pred
mut_pred

mal_conf = max(p_mal, 1-p_mal)
mut_conf = max(p_mut, 1-p_mut)
```

只保留：

```python
mal_pred == mal_label
and
mut_pred == mut_label
and
mal_conf >= conf_threshold
and
mut_conf >= conf_threshold
```

例如：

```python
conf_threshold = 0.65
```

这些样本组成：

```python
high_confidence_pool
```

原因：

希望 synthetic samples 来自已经被模型正确理解的样本，
而不是来自噪声样本或困难样本。

---

## 动态特征空间增强

对于每个需要增强的少数类：

例如：

```python
BM
MM
```

从 high_confidence_pool 中选择对应类别样本。

随机选择以下方法之一：

```python
mixup
borderline_smote
prototype
```

生成 synthetic features。

要求：

每个 synthetic sample 独立随机选择方法。

例如：

```python
50% mixup
25% borderline_smote
25% prototype
```

或者均匀随机。

---

## Synthetic-only Tomek Links

生成 synthetic features 后：

执行 Tomek Links。

定义保持标准定义：

```python
mutual nearest neighbors
different classes
```

但删除策略修改为：

仅删除 synthetic samples。

绝不删除真实样本。

例如：

```python
real samples:
0 ... n_real-1

synthetic samples:
n_real ... N-1
```

若 Tomek Link 涉及 synthetic sample：

删除 synthetic sample。

若 Tomek Link 仅涉及真实样本：

保留。

---

## 动态更新频率

不要每个 epoch 都重新生成 synthetic samples。

增加参数：

```python
update_synthetic_every = 3
```

即：

```python
epoch 15
生成一次

epoch 18
重新生成

epoch 21
重新生成
```

中间 epoch 继续使用当前 synthetic pool。

原因：

避免训练目标剧烈波动。

---

## 训练目标

真实图像始终参与训练。

Synthetic feature 不替代真实样本。

训练损失：

```python
L_total
=
L_real
+
lambda_syn * L_synthetic
```

其中：

```python
lambda_syn = 0.2 ~ 0.5
```

要求：

真实图像分支继续更新：

* backbone
* classifier

Synthetic feature 分支：

只更新 classifier head。

不要让 synthetic feature 直接驱动 backbone 更新。

---

## 需要实现的内容

请修改代码实现：

1. Dynamic feature pool manager
2. Activation trigger
3. High-confidence sample selection
4. Dynamic synthetic generation
5. Synthetic-only Tomek Links cleaning
6. Periodic regeneration
7. Synthetic loss branch
8. Configurable hyperparameters

新增建议参数：

```python
--dynamic_oversampling
--activation_threshold
--activation_patience
--conf_threshold
--update_synthetic_every
--lambda_syn
--synthetic_ratio
--tomek_links
```

---

## 日志输出

每次 synthetic pool 更新时打印：

```text
Epoch XX

Dynamic oversampling activated

High-confidence samples:
BN=...
BM=...
MN=...
MM=...

Synthetic generated:
mixup=...
borderline_smote=...
prototype=...

Tomek removed:
...

Final synthetic pool:
...
```

并保存 CSV 记录。

---

## 要求

请给出完整可运行代码修改方案。

不要只给伪代码。

请指出：

* 哪些函数需要新增；
* 哪些函数需要修改；
* 训练循环如何改造；
* 数据流如何变化；
* 如何避免内存泄漏和重复计算。
