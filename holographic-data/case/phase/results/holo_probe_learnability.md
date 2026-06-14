# Holographic nested_monoid — GPU learnability probe ✅

单 cell (β=0.5, γ=0.2)，holo_small (RoPE, 19M)，2000 steps，gpu_test。

| eval length | answer accuracy |
|---|---|
| 256 (train) | **1.000** |
| 512 | 0.708 |
| 1024 | 0.370 |
| 2048 | 0.193 |

**结论**：任务在训练长度**完全可学**（acc=1.0），retention 在真实信号上测量（非噪声）；
并呈现优雅的长度泛化衰减（8× 长度 1.0→0.19）——正是相图要刻画的现象。✅ 方法验证通过，可铺开 campaign。
