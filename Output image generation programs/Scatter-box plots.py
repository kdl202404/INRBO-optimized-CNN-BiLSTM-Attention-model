import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams
#
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
rcParams['axes.unicode_minus'] = False    # 正确显示负号

font_path = r"C:\Windows\Fonts\simhei.ttf"
my_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False


# === 读取数据（记得修改文件路径）
df_raw = pd.read_excel('D:\Desktop\缝合\程序\箱线-散点图数据.xlsx')

# 第一列是真实值
y_true = df_raw.iloc[:, 0].values

# 第二到第十一列是预测值
predictions = df_raw.iloc[:, 1:]

# 模型名称
model_names = predictions.columns.tolist()

# === 计算每个模型的偏差（%）
data = []
for model in model_names:
    y_pred = predictions[model].values
    deviation = (y_pred - y_true) / y_true * 100  # 偏差百分比
    for dev in deviation:
        data.append({'Model': model, 'Deviation': dev})

# 转成DataFrame
df = pd.DataFrame(data)

# === 设置颜色调色板（为每个模型选择不同颜色）
palette = sns.color_palette("Set2", len(model_names))  # 选择颜色调色板

# === 开始绘图
plt.figure(figsize=(14, 8))

# 小提琴图 + 箱线图
sns.violinplot(
    x='Model', y='Deviation', data=df,
    inner=None,  # 去掉内部的小图形
    width=0.5,   # 控制小提琴的宽度
    scale='count',  # 根据数据的分布调整宽度
    palette=palette,  # 使用自定义的颜色调色板
    ax=plt.gca(),  # 确保绘制在当前轴上
    alpha=0.6  # 调整透明度
)

sns.boxplot(
    x='Model', y='Deviation', data=df,
    width=0.2,
    showfliers=False,
    boxprops={'facecolor':'none', 'edgecolor':'black'},
    whiskerprops={'color':'black'},
    capprops={'color':'black'},
    medianprops={'color':'red', 'linewidth':2},
    palette=palette,  # 使用自定义的颜色调色板
    ax=plt.gca()  # 确保绘制在当前轴上
)

# 散点图（添加散点）
sns.stripplot(
    x='Model', y='Deviation', data=df,
    color='blue', size=6, jitter=0.25, alpha=0.3,
    ax=plt.gca()  # 确保绘制在当前轴上
)

# === 美化图表
# plt.title('Deviation of Predicted Values by Model', fontsize=16, fontweight='bold')
# plt.ylabel('Deviation (%)', fontsize=14)
# plt.xlabel('Models', fontsize=14)

# plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.title('模型预测值偏差', fontsize=18, fontweight='bold')
plt.ylabel('相对偏差 (%)', fontsize=14)
plt.xlabel('模型', fontsize=15)
plt.xticks(fontsize=13, rotation=45)
plt.yticks(fontsize=13)#ticks=np.arange(-50, 51, 10),y轴刻度间隔设置
plt.ylim(-50, 50)

plt.gca().set_facecolor('white')  # 设置背景颜色为淡灰色
plt.tight_layout()  # 自动调整布局

# 替换 plt.show() 后的保存部分为：
save_path = r"D:\Desktop\箱线-散点图.jpg"  # 修改为你想保存的路径
plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 先保存
plt.show()  # 再显示


# === 输出每个模型偏差的标准差
print("各模型偏差的标准差（%）:")
for model in model_names:
    y_pred = predictions[model].values
    deviation = (y_pred - y_true) / y_true * 100
    std_dev = np.std(deviation, ddof=1)  # 使用样本标准差（ddof=1）
    print(f"{model}: {std_dev:.3f}%")
