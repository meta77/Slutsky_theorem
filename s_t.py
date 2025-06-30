import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set(style='whitegrid')

# サンプルサイズ
n_values = [10, 100, 10000, 100000]
num_samples = 10000

# 結果を可視化する
fig, axes = plt.subplots(1, len(n_values), figsize=(16, 4))

for i, n in enumerate(n_values):
    # X_n ~ N(0, 1/n)
    X_n = np.random.normal(loc=0, scale=1/np.sqrt(n), size=num_samples) # 生成する乱数列の要素数（サンプル数）を num_samples に指定。

    # Y_n ~ 1 + noise（確率収束する定数）
    Y_n = 1 + np.random.normal(loc=0, scale=1/np.sqrt(n), size=num_samples)

    # Z_n = X_n * Y_n（Slutskyの対象）
    Z_n = X_n * Y_n

    # ヒストグラムと理論分布を描画
    ax = axes[i]
    sns.histplot(Z_n, kde=True, stat="density", ax=ax, bins=40, color='skyblue', label=f"n={n}")

    # 理論分布（N(0, 1/n)）に近づくか？
    x = np.linspace(-0.2, 0.2, 200)
    ax.plot(x, norm.pdf(x, loc=0, scale=1/np.sqrt(n)), color='red', linestyle='--', label='Theoretical N(0, 1/n)')

    ax.set_title(f"n = {n}")
    ax.legend()

plt.suptitle("Z_n = X_n * Y_n → N(0, 1/n)", fontsize=14)
plt.tight_layout()
plt.show()
