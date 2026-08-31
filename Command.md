请修改我的 `train_head_mlp.py`，重点修正 Tomek Links cleaning 的实现。

当前代码已经有：

* `feature_space_oversample(...)`
* `tomek_links_clean(...)`
* 在 oversampling 后通过 `--tomek_links` 调用 Tomek cleaning

但现在的 Tomek Links 删除策略不合适：它会从完整训练集里删除 majority-side 样本，可能删除真实训练样本。对于医学小样本任务，我只希望 Tomek Links 用来清理“合成样本”，不要删除任何真实样本。

请按以下要求修改：

1. 在 oversampling 前记录真实训练样本数量：

```python
n_real_train = len(train_feat)
```

2. 将现有 `tomek_links_clean(...)` 替换为 synthetic-only 版本，例如：

```python
def tomek_links_clean(features, mal_labels, mut_labels, n_real):
    ...
```

3. Tomek Link 定义保持不变：

   * 对所有样本建立 `NearestNeighbors(n_neighbors=2)`
   * 对每个样本找最近邻
   * 如果 `i` 和 `j` 互为最近邻，并且 `c4_labels[i] != c4_labels[j]`，则它们构成 Tomek Link

4. 删除规则改为：

   * 如果 Tomek Link 中 `i >= n_real`，删除 `i`
   * 如果 Tomek Link 中 `j >= n_real`，删除 `j`
   * 永远不要删除 `i < n_real` 或 `j < n_real` 的真实样本
   * 不要再根据 class_counts 删除 majority-side

5. 调用处改为：

```python
if args.tomek_links:
    train_feat, train_mal, train_mut, n_removed = tomek_links_clean(
        train_feat, train_mal, train_mut, n_real=n_real_train
    )
```

6. 保留 `--tomek_links` 参数，不改变其它训练逻辑。

7. 打印清理信息，例如：

   * 总样本数
   * 真实样本数
   * 合成样本数
   * 删除的 synthetic 样本数
   * 删除比例

8. 注意边界情况：

   * 如果 `len(features) < 2`，直接返回
   * 如果没有 Tomek Links，返回原数组
   * 如果没有合成样本，即 `n_real >= len(features)`，直接返回

9. 不要修改 validation / calibration / test 集。

10. 不要引入 `imblearn`，继续使用 `sklearn.neighbors.NearestNeighbors`。

请输出完整修改后的相关代码片段，包括：

* 新的 `tomek_links_clean` 函数
* oversampling 前后调用位置的修改

