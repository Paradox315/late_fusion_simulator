import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment

config = {
    "font.family": "serif",  # 衬线字体
    "font.serif": ["SimSun"],  # 宋体
    "mathtext.fontset": "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "axes.unicode_minus": False,  # 处理负号，即-号
}
plt.rcParams.update(config)

# Load distance matrices
dist_data = pd.read_csv("./dist_mat.csv").values  # 加载距离矩阵
dist_data_noise = pd.read_csv("./dist_mat_noise.csv").values

# 绘制并保存原始距离矩阵热力图
plt.figure(figsize=(10, 8))

hm1 = sns.heatmap(
    dist_data, cmap="coolwarm", annot_kws={"size": 20}, fmt=".2f", annot=True
)
plt.xticks(ticks=[], labels=[])
plt.yticks(ticks=[], labels=[])
plt.tight_layout()
plt.savefig("fig1-cost1.pdf", bbox_inches="tight")
plt.close()

# 绘制并保存噪声距离矩阵热力图
plt.figure(figsize=(10, 8))
hm2 = sns.heatmap(
    dist_data_noise, cmap="coolwarm", annot_kws={"size": 20}, fmt=".2f", annot=True
)
plt.xticks(ticks=[], labels=[])
plt.yticks(ticks=[], labels=[])
plt.tight_layout()
plt.savefig("fig1-cost2.pdf", bbox_inches="tight")
plt.close()

min_dist = 0.5
row_ind, col_ind = linear_sum_assignment(-dist_data)

mask = dist_data[row_ind, col_ind] > min_dist
true_matches = list(zip(row_ind[mask], col_ind[mask]))


# Get assignments from the noisy matrix
row_ind_noise, col_ind_noise = linear_sum_assignment(dist_data_noise)
noise_matches = list(zip(row_ind_noise, col_ind_noise))

# Identify correct and incorrect matches
correct_matches = set(true_matches).intersection(set(noise_matches))
incorrect_matches = set(noise_matches) - set(true_matches)

# Get distance values for each category
true_match_values = [dist_data[i, j] for i, j in true_matches]
true_match_noisy_values = [dist_data_noise[i, j] for i, j in true_matches]
incorrect_match_values = [dist_data[i, j] for i, j in incorrect_matches]
incorrect_match_noisy_values = [dist_data_noise[i, j] for i, j in incorrect_matches]

# Create a figure with four subplots
# Plot distributions
plt.figure(figsize=(16, 9))
sns.histplot(true_match_values, kde=True, color="green")
plt.xlabel("相似度", fontsize=30)
plt.ylabel("频数", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig("fig3-cost1.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(16, 9))
sns.histplot(true_match_noisy_values, kde=True, color="blue")
plt.xlabel("相似度", fontsize=30)
plt.ylabel("频数", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig("fig3-cost2.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(16, 9))
sns.histplot(incorrect_match_values, kde=True, color="orange")
plt.xlabel("相似度", fontsize=30)
plt.ylabel("频数", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig("fig3-cost3.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(16, 9))
sns.histplot(incorrect_match_noisy_values, kde=True, color="red")
plt.xlabel("相似度", fontsize=30)
plt.ylabel("频数", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig("fig3-cost4.pdf", bbox_inches="tight")
plt.close()


# Adjust layout and save

# Print statistics
print(f"Total matches: {len(true_matches)}")
print(
    f"Correct matches preserved: {len(correct_matches)} ({len(correct_matches)/len(true_matches)*100:.2f}%)"
)
print(
    f"Incorrect matches: {len(incorrect_matches)} ({len(incorrect_matches)/len(true_matches)*100:.2f}%)"
)
print("\nAverage distances:")
print(f"True matches in original matrix: {np.mean(true_match_values):.4f}")
print(f"True matches in noisy matrix: {np.mean(true_match_noisy_values):.4f}")
print(f"Incorrect matches in original matrix: {np.mean(incorrect_match_values):.4f}")
print(f"Incorrect matches in noisy matrix: {np.mean(incorrect_match_noisy_values):.4f}")
