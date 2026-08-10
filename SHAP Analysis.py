import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn.ensemble import RandomForestRegressor
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
rcParams['axes.unicode_minus'] = False    # 正确显示负号
# 获取当前工作目录
current_directory = os.getcwd()
print(f"当前工作目录: {current_directory}")

warnings.filterwarnings('ignore')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码问题
# plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题


# import constants

read_path = r'D:\Desktop\缝合\主喷嘴\shap分析\shap分析数据.xlsx'
# read_sheet = '出图'
# 读取数据集
dataset = pd.read_excel(read_path)  # 读取数据
values = dataset.values[:, 0:]
values = np.array(values)

# 划分训练集、验证集、测试集
X = values[:, :-1]  # 特征
Y = values[:, -1]  # 目标

# 获取特征名称
feature_names = dataset.columns[:-1].tolist()  # 自动获取特征名称
# 获取目标变量的列名（假设目标变量是最后一列）
output_name = dataset.columns[-1]

print(f"输出的特征名称: {output_name}")

print(f"特征名称: {feature_names}")

# 随机森林回归模型
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X, Y)

# 使用 SHAP 来解释模型
explainer = shap.TreeExplainer(model_rf)  # 使用 TreeExplainer 来解释树模型
shap_values = explainer.shap_values(X)  # 计算 SHAP 值

# 创建保存路径 'shap_png' 文件夹（如果不存在）
# output_dir = 'shap_png0428/' + read_sheet
output_dir =r'D:\Desktop\缝合\主喷嘴\shap分析\出图6'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 绘图但不显示
shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)

# 设置横纵坐标
# plt.xlabel("SHAP value(impact on model output)")
plt.xlabel("SHAP值（对模型输出的影响）")
# 获取 color bar（最后一个 axes 是 colorbar）
cbar = plt.gcf().axes[-1]  # 获取图中的最后一个 axes，即 color bar

# 修改 colorbar 的标签
# cbar.set_ylabel("Feature value", fontsize=12)  # 修改 colorbar 的标题
# cbar.set_yticklabels(["Low", "High"])  # 将 colorbar 的 tick labels 修改为中文
cbar.set_ylabel("特征值", fontsize=12)  # 修改 colorbar 的标题
cbar.set_yticklabels(["低", "高"])  # 将 colorbar 的 tick labels 修改为中文
# 保存图片
summary_plot_path = os.path.join(output_dir, 'shap_summary_plot.png')
plt.savefig(summary_plot_path, dpi=500, bbox_inches='tight')
plt.close()


'''
横坐标（X轴）：
特征 i 的原始值（仍然是输入数据中该特征的实际取值）。

纵坐标（Y轴）：
特征 i 对预测值的 SHAP 值（贡献）。
颜色（color）：
与特征 j 的值相关（默认颜色条是 Viridis 颜色映射）。
→ 表示 i 的 SHAP 值是否受到 j 的交互影响。
'''
for i in feature_names:
    for j in feature_names:
        if (i == j):
            continue

        shap.dependence_plot(i, shap_values, X, feature_names=feature_names, interaction_index=j,
                             show=False)
        plt.xlabel(i)  # 修改横坐标
        plt.ylabel("SHAP value(" + output_name + ")")  # 修改纵坐标
        plot_path = os.path.join(output_dir, f'shap_{i}_vs_{j}.png')
        print(plot_path)
        plt.savefig(plot_path, dpi=500)
        plt.close()
        print(f"两两特征依赖图已保存为 {plot_path}")


'''
横坐标（X轴）：
特征 i 的原始值（原始输入数据中这一列的取值）。
→ 反映该特征在样本中的实际值。

纵坐标（Y轴）：
特征 i 对预测结果的 SHAP值。
→ 表示该特征的某个具体取值对模型预测结果的“正向”或“负向”贡献有多大。
'''
for i in feature_names:
    shap.dependence_plot(i, shap_values, X, feature_names=feature_names, interaction_index=None,
                         show=False)
    plt.xlabel(i)  # 设置横坐标
    plt.ylabel("SHAP value(" + output_name + ")")  # 设置纵坐标
    plot_path = os.path.join(output_dir, f'shap_{i}.png')
    print(plot_path)
    plt.savefig(plot_path, dpi=500)
    plt.close()
    print(f"单一特征依赖图已保存为 {plot_path}")

if isinstance(shap_values, list):
    shap_values = shap_values[0]

# 将 SHAP 值转换为 Explanation 对象，确保与 bar() 方法兼容
shap_exp = shap.Explanation(shap_values, feature_names=feature_names)

# 绘制 SHAP 全局特征重要性条形图
plt.figure()
shap.plots.bar(shap_exp, show=False)

# 修改横纵坐标
# plt.xlabel("平均 SHAP 值")
# plt.ylabel("特征名称")
# plt.xlabel("Average SHAP value", fontsize=12)
# plt.ylabel("Feature name", fontsize=12)
plt.xlabel("平均SHAP值", fontsize=12)
plt.ylabel("特征名", fontsize=12)
# 保存图像
bar_plot_path = os.path.join(output_dir, 'shap_global_bar_plot.png')
plt.savefig(bar_plot_path, dpi=500, bbox_inches='tight')
plt.close()
print(f"SHAP 全局特征重要性图已保存为 {bar_plot_path}")



#相关性热图
import seaborn as sns

# 创建一个 DataFrame 来保存 SHAP 值（行是样本，列是特征）
df_shap = pd.DataFrame(shap_values, columns=feature_names)
# 计算 SHAP 值的相关性矩阵（可以使用 spearman 也可以用 pearson）
shap_corr_matrix = df_shap.corr(method='spearman')
# 绘制相关性热图
plt.figure(figsize=(12, 10))
sns.heatmap(shap_corr_matrix,
            annot=True,        # 显示相关系数数值
            fmt=".2f",         # 保留2位小数
            cmap='coolwarm',   # 配色方案
            linewidths=0.5,    # 网格线
            square=True,       # 保持方格
            # vmin=-1,  # 添加此行
            # vmax=1,  # 添加此行
            cbar_kws={"shrink": .8},  # 缩小 color bar
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot_kws={"size": 14})

# 设置标题和标签字体
# plt.title('基于 SHAP 值的特征相关性热图', fontsize=16)
plt.title('Feature correlation heat map based on SHAP value', fontsize=16)
plt.xticks(rotation=45, ha='right',fontsize=14)
plt.yticks(rotation=0,fontsize=14)
# 保存图像
shap_corr_plot_path = os.path.join(output_dir, 'shap_feature_correlation_heatmap.png')
plt.tight_layout()
plt.savefig(shap_corr_plot_path, dpi=500)
plt.close()
print(f"SHAP 特征相关性热图已保存为 {shap_corr_plot_path}")

