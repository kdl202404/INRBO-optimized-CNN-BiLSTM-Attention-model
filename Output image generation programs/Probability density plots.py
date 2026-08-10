import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, linregress

# === 1. 读取 Excel 数据 ===
filename = r'D:\Desktop\1\4. cnn_att_svr_aoa_train.xlsx'  # 可换成测试集文件
data = pd.read_excel(filename)

y_true = data['True'].values
y_pred = data['Predicted'].values

# === 2. 计算二维密度 ===
xy = np.vstack([y_true, y_pred])
density = gaussian_kde(xy)(xy)

# === 3. 散点密度图 ===
plt.figure(figsize=(8, 6))
sc = plt.scatter(y_true, y_pred, c=density, s=30, edgecolors='k', cmap='jet', alpha=0.8)

# # === 4. 添加颜色条 ===
from matplotlib.ticker import ScalarFormatter, MaxNLocator
# 添加 colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Scatter Density', fontsize=12)
# 设置刻度为整数
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()
# 设置格式为整数 + 科学计数法（指数自动确定）
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # 始终使用科学计数法
formatter.set_scientific(True)
cbar.ax.yaxis.set_major_formatter(formatter)
# 调整指数部分字体
cbar.ax.yaxis.offsetText.set_fontsize(10)

# === 5. 添加对角线 ===
min_val = min(np.min(y_true), np.min(y_pred)) - 0.05
max_val = max(np.max(y_true), np.max(y_pred)) + 0.05
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.3)

# === 6. 线性回归 ===
slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
y_fit = slope * y_true + intercept
plt.plot(y_true, y_fit, 'r-', linewidth=1.5)

# === 7. 计算统计指标 ===
N = len(y_true)
R = r_value
BIAS = np.mean(y_pred - y_true)
RMSE = np.sqrt(np.mean((y_pred - y_true)**2))

# === 8. 添加统计信息文本 ===
text_str = f'N = {N}\nR = {R:.2f}\nBIAS = {BIAS:.2f}\nRMSE = {RMSE:.2f}'
# plt.text(min_val + 5.0, max_val - 10.0, text_str, fontsize=10, color='k', fontweight='bold')
plt.text(0.05, 0.86, text_str, fontsize=10, color='k', fontweight='bold',
         transform=plt.gca().transAxes, horizontalalignment='left', verticalalignment='bottom')

# === 9. 图像美化 ===
plt.title('Probability Density', fontsize=12, fontweight='bold')
plt.xlabel('Actual value (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted value (m/s)', fontsize=12, fontweight='bold')
plt.grid(False)
plt.xlim([np.min(y_true), np.max(y_true)])
plt.ylim([np.min(y_pred), np.max(y_pred)])
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().tick_params(labelsize=12, width=1.5)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gcf().patch.set_facecolor('white')  # 背景设白
plt.tight_layout()
plt.show()
