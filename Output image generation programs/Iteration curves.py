# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取四个Excel文件
# curve1_df = pd.read_csv(r'D:\Desktop\缝合\主喷嘴\迭代和箱线出图数据\1\迭代数据\HEOA_Curve.csv')  # 替换为你的文件名
# curve2_df = pd.read_csv(r'D:\Desktop\缝合\主喷嘴\迭代和箱线出图数据\1\迭代数据\SCNGO_Curve.csv')
# curve3_df = pd.read_csv(r'D:\Desktop\缝合\主喷嘴\迭代和箱线出图数据\1\迭代数据\NRBO_Curve.csv')
# curve4_df = pd.read_csv(r'D:\Desktop\缝合\主喷嘴\迭代和箱线出图数据\1\迭代数据\INRBO_Curve.csv')
#
# # 提取Fitness列（默认按行顺序绘图）
# curve1 = curve1_df['Fitness'].values
# curve2 = curve2_df['Fitness'].values
# curve3 = curve3_df['Fitness'].values
# curve4 = curve4_df['Fitness'].values
#
# # 存储所有曲线数据（算法名, 曲线数据）
# all_curve_data = [
#     ('HEOA', curve1),
#     ('SCNGO', curve2),
#     ('NRBO', curve3),
#     ('INRBO', curve4)
# ]
# # 绘图
# plt.figure(figsize=(10, 6))
# for algorithm, curve_data in all_curve_data:
#     plt.plot(range(len(curve_data)), curve_data, label=algorithm)
# plt.xlabel('The number of iterations', fontsize=12)
# plt.ylabel('RMSE', fontsize=12)
# plt.title('Convergence curve of algorithm', fontsize=12)
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()
# # 替换 plt.show() 后的保存部分为：
# save_path = r"D:\Desktop\迭代图.jpg"  # 修改为你想保存的路径
# plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 先保存
# plt.show()  # 再显示




import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import font_manager

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
rcParams['axes.unicode_minus'] = False    # 正确显示负号

font_path = r"C:\Windows\Fonts\simhei.ttf"
my_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False
# rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
# rcParams['axes.unicode_minus'] = False    # 正确显示负号
# 读取四个Excel文件
curve1_df = pd.read_csv(r'D:\Desktop\缝合\主喷嘴\迭代和箱线出图数据\1\迭代数据\HEOA_Curve.csv')  # 替换为你的文件名
curve2_df = pd.read_csv(r'D:\Desktop\缝合\主喷嘴\迭代和箱线出图数据\1\迭代数据\SCNGO_Curve.csv')
curve3_df = pd.read_csv(r'D:\Desktop\缝合\主喷嘴\迭代和箱线出图数据\1\迭代数据\NRBO_Curve.csv')
curve4_df = pd.read_csv(r'D:\Desktop\缝合\主喷嘴\迭代和箱线出图数据\1\迭代数据\INRBO_Curve.csv')

# 提取Fitness列（默认按行顺序绘图）
curve1 = curve1_df['Fitness'].values
curve2 = curve2_df['Fitness'].values
curve3 = curve3_df['Fitness'].values
curve4 = curve4_df['Fitness'].values

# 存储所有曲线数据（算法名, 曲线数据）
all_curve_data = [
    ('HEOA', curve1),
    ('SCNGO', curve2),
    ('NRBO', curve3),
    ('INRBO', curve4)
]
# 定义颜色、线型、marker，确保四条曲线明显区分
colors = ['purple', 'blue', 'green', 'red']
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'd']

plt.figure(figsize=(10, 6))
for (algorithm, curve_data), color, ls, mk in zip(all_curve_data, colors, linestyles, markers):
    plt.plot(range(len(curve_data)), curve_data,
             label=algorithm,
             color=color,
             linestyle=ls,
             marker=mk,
             markevery=max(1, len(curve_data)//20),  # 控制 marker 数量，避免太密
             linewidth=1.8)

plt.xlabel('迭代次数', fontsize=14)
plt.ylabel('均方根误差', fontsize=14)
plt.title('算法收敛曲线', fontsize=16)
# plt.xlabel('The number of iterations', fontsize=14)
# plt.ylabel('RMSE', fontsize=14)
# plt.title('Convergence curve of algorithm', fontsize=16)

# 坐标刻度字体大小和颜色
plt.xticks(fontsize=13, color='black')
plt.yticks(fontsize=13, color='black')

plt.legend(
    fontsize=14,        # 字体大小
    title_fontsize=13,  # 图例标题字体（如果有标题）
    markerscale=2,    # 调整 marker 大小
)

plt.tight_layout()

save_path = r"D:\Desktop\迭代图.jpg"
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

