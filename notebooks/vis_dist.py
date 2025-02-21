import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")
dist_mat = pd.read_csv("dist_mat.csv", index_col=0)
dist_mat = dist_mat.values
sns.heatmap(dist_mat)
plt.show()
dist_mat_noise = pd.read_csv("dist_mat_noise.csv", index_col=0)
dist_mat_noise = dist_mat_noise.values
sns.heatmap(dist_mat_noise)
plt.show()
