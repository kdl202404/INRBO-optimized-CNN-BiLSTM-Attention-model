# 调用相关库
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable

# 读取数据
dataset = pd.read_excel(r"D:\Desktop\缝合\主喷嘴\数据\Vx 2.xlsx",
                        sheet_name="输出参数-总表",
                        skiprows=1)
# dataset= dataset.head(1500)
print(dataset)
values = dataset.values[:, 1:]

# 分割数据集
# 筛选掉输出值小于240的数据
filtered_values = values[values[:, -1] >= 100]  # 保留输出值大于等于240的数据
num_samples = filtered_values.shape[0]
print(num_samples)
np.random.seed(42)  # 42可以替换成你喜欢的任意整数
per = np.random.permutation(num_samples)

n_train_number = per[:int(num_samples * 0.8)]
n_test_number = per[int(num_samples * 0.955):]

Xtrain = filtered_values[n_train_number, :-1]
Ytrain = filtered_values[n_train_number, -1].reshape(-1, 1)
Xtest = filtered_values[n_test_number, :-1]
Ytest = filtered_values[n_test_number, -1].reshape(-1, 1)

# 数据归一化
m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)
vp_test = m_in.transform(Xtest)

m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)
vt_test = m_out.transform(Ytest)

# 转换为 PyTorch tensors
vp_train = torch.tensor(vp_train, dtype=torch.float32).unsqueeze(1)   # 增加时间步维度
vt_train = torch.tensor(vt_train, dtype=torch.float32)

vp_test = torch.tensor(vp_test, dtype=torch.float32).unsqueeze(1)
vt_test = torch.tensor(vt_test, dtype=torch.float32)


# 定义 CNN-BiLSTM 模型
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1)
        self.maxpooling = nn.MaxPool1d(kernel_size=1)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, 1)  # BiLSTM有双向，输出是hidden_size的两倍

    def forward(self, x):
        # 先通过卷积层，卷积操作需要输入形状为[batch_size, 输入通道数, seq_len]
        x = x.permute(0, 2, 1)  # 将输入从[batch_size, seq_len, 输入通道数]调整为[batch_size, 输入通道数, seq_len]
        x = self.conv1d(x)
        x = self.maxpooling(x)

        # 卷积后的输出形状调整为LSTM所需的形状 [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)

        # LSTM层
        _, (hn, _) = self.bilstm(x)

        # LSTM的输出 hn 是 [num_layers * num_directions, batch_size, hidden_size]
        # 将其调整为 [batch_size, hidden_size * num_directions]
        hn = hn.permute(1, 0, 2).reshape(hn.shape[1], -1)

        # 全连接层
        out = self.fc(hn)
        return out


# 实例化模型，定义损失函数和优化器
model = CNN_BiLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
n_epochs = 100
batch_size = 64
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    permutation = torch.randperm(vp_train.size()[0])
    train_loss = 0.0
    for i in range(0, vp_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = vp_train[indices], vt_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_losses.append(train_loss / (vp_train.size()[0] // batch_size))

    # 验证集损失
    model.eval()
    with torch.no_grad():
        val_outputs = model(vp_test)
        val_loss = criterion(val_outputs, vt_test)
        val_losses.append(val_loss.item())

    print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss.item():.4f}')

# 绘制损失曲线
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss', linestyle='-', marker='o', markersize=2)
plt.plot(val_losses, label='Validation Loss', linestyle='-', marker='x', markersize=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# 预测与评估
model.eval()
with torch.no_grad():
    yhat = model(vp_test).numpy()
    predicted_data = m_out.inverse_transform(yhat)
    yhat_train = model(vp_train).numpy()
    predicted_data_train = m_out.inverse_transform(yhat_train)
#
# # 定义评估函数
# def mape(y_true, y_pred):
#     non_zero_indices = y_true != 0
#     if np.sum(non_zero_indices) == 0:
#         return float('inf')
#     return np.mean(np.abs((y_pred[non_zero_indices] - y_true[non_zero_indices]) / y_true[non_zero_indices])) * 100


# 定义 IA 指数的计算函数
def index_of_agreement(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_true - mean_y_true) + np.abs(y_pred - mean_y_true)) ** 2)
    return 1 - (numerator / denominator)

# 定义 TIC 指数的计算函数
def theils_inequality_coefficient(y_true, y_pred):
    numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    return numerator / denominator


def evaluate_forecasts(Ytest, predicted_data):
    # 计算 MAE、RMSE、IA、TIC
    mae = mean_absolute_error(Ytest, predicted_data)
    mse = mean_squared_error(Ytest, predicted_data)
    rmse = np.sqrt(mse)
    ia = index_of_agreement(Ytest, predicted_data)
    tic = theils_inequality_coefficient(Ytest, predicted_data)
    r2 = r2_score(Ytest, predicted_data)
    return  rmse, mae, ia,tic, r2

rmse, mae, ia,tic, r2 = evaluate_forecasts(Ytest, predicted_data)

# 输出评估结果
table = PrettyTable(['测试集指标', 'RMSE', 'MAE','IA','TIC', 'R2'])
table.add_row(['预测结果指标：', rmse, mae, ia,tic, f'{r2 * 100}%'])
print(table)
metrics = {
    'Metric': ['R²', 'MAE', 'RMSE','IA','TIC'],
    'Value': [r2, mae, rmse, ia,tic]
}
df_metrics = pd.DataFrame(metrics)
plt.figure(figsize=(6, 2))
# plt.title("CNN-BiLSTM-Attention")
plt.axis('tight')
plt.axis('off')
table = plt.table(cellText=df_metrics.values, colLabels=df_metrics.columns, cellLoc='center', loc='center')
table.scale(1, 2)  # 调整表格的大小
table.auto_set_font_size(False)
table.set_fontsize(12)
plt.show()


# 绘制结果图
plt.figure(figsize=(8, 6))
plt.plot(predicted_data, label='Predicted value', linestyle='--', marker='o',
         color='blue', linewidth=2)
plt.plot(Ytest, label='Actual value', linestyle='-', marker='x',
         color=(223/255, 143/255, 120/255), linewidth=2)

plt.xlabel("Sample points", fontsize=12)
plt.ylabel("Airflow Velocity (m/s)", fontsize=12)

# -------------------- 坐标轴范围 --------------------
plt.ylim(bottom=60, top=320)

# -------------------- 刻度样式（重点新增） --------------------
tick_fontsize = 12   # ← 刻度数字大小（可调）
tick_width = 1.0     # ← 刻度线粗细（可调）
tick_length = 5      # ← 刻度线长度（可调）

plt.tick_params(axis='x', which='both',
                labelsize=tick_fontsize,
                width=tick_width,
                length=tick_length,
                direction='in')

plt.tick_params(axis='y', which='both',
                labelsize=tick_fontsize,
                width=tick_width,
                length=tick_length,
                direction='in')

# # 刻度数字加粗
# ax = plt.gca()
# for label in ax.get_xticklabels():
#     label.set_fontweight('bold')
# for label in ax.get_yticklabels():
#     label.set_fontweight('bold')
#
# # 坐标轴边框加粗（论文风格）
# for spine in ax.spines.values():
#     spine.set_linewidth(2)

# -------------------- 图例 --------------------
plt.legend(
    loc='lower left',
    bbox_to_anchor=(0.70, 0.88),
    ncol=1,
    labelspacing=0.4,
    frameon=True,
    framealpha=0.8,
    edgecolor='#666666',
    facecolor='#F5F5F5',
    fontsize=12
)

plt.tight_layout()

# -------------------- 保存 --------------------
save_path = r"D:\Desktop\c-l.jpg"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# # 保存测试集的真实值和预测值到 Excel 文件
# result_df = pd.DataFrame({
#     'Actual Value': Ytest.flatten(),
#     'Predicted Value': predicted_data.flatten()
# })
#
# # 保存为 Excel 文件
# output_file = r"D:\Desktop\CNN_BiLSTM预测结果.xlsx"
# result_df.to_excel(output_file, index=False)
# print(f"预测结果已保存到：{output_file}")





# from plots import plot_all
# import numpy as np
# import pandas as pd
# # 将 Ytest 和 predicted_data 转换为浮点数类型
# Ytest = np.array(Ytest, dtype=np.float64)
# predicted_data = np.array(predicted_data, dtype=np.float64)
# Ytrain = np.array(Ytrain, dtype=np.float64)
# predicted_data_train = np.array(predicted_data_train, dtype=np.float64)
# # 检查是否有 NaN 或无穷大
# print(np.isnan(Ytest).sum(), np.isnan(predicted_data).sum())  # 打印 NaN 的数量
# print(np.isinf(Ytest).sum(), np.isinf(predicted_data).sum())  # 打印无穷大的数量
# print(np.isnan(Ytrain).sum(), np.isnan(predicted_data_train).sum())  # 打印 NaN 的数量
# print(np.isinf(Ytrain).sum(), np.isinf(predicted_data_train).sum())  # 打印无穷大的数量
# # 替换 NaN 或无穷大的值为 0（或其他合适的值）
# Ytest = np.nan_to_num(Ytest, nan=0.0, posinf=0.0, neginf=0.0)
# predicted_data = np.nan_to_num(predicted_data, nan=0.0, posinf=0.0, neginf=0.0)
# Ytrain = np.nan_to_num(Ytrain, nan=0.0, posinf=0.0, neginf=0.0)
# predicted_data_train = np.nan_to_num(predicted_data_train, nan=0.0, posinf=0.0, neginf=0.0)
# # 检查形状是否匹配
# print(Ytest.shape, predicted_data.shape)
# print(Ytrain.shape, predicted_data_train.shape)
# # 如果不匹配，可以进行必要的调整，例如：
# Ytest = Ytest.reshape(predicted_data.shape)
# Ytrain = Ytrain.reshape(predicted_data_train.shape)
# plot_all(Ytrain,predicted_data_train,Ytest, predicted_data)

#
#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 示例数据（请替换为您的 y_test 和 y_test_predicted 数据）
# # 将 Ytest 和 predicted_data 转换为浮点数类型并进行一维化
# Ytest = np.array(Ytest, dtype=np.float64).flatten()
# predicted_data = np.array(predicted_data, dtype=np.float64).flatten()
#
# # 创建 DataFrame
# data = pd.DataFrame({
#     'Actual': Ytest,
#     'Predicted': predicted_data
# })
#
#
# # 转换为长格式，便于绘图
# data_melted = data.melt(var_name='Type', value_name='Value')
#
# # 设置风格和调色板
# sns.set(style="whitegrid")
#
# # 设置图像大小和分辨率
# plt.figure(dpi=120)
#
# # 绘制箱线图
# sns.boxplot(x='Type', y='Value', data=data_melted, palette="deep")
#
# # 设置标题和轴标签
# plt.title('Boxplot of Actual and Predicted Values', fontsize=16, fontweight='bold')
# plt.xlabel('Value Type', fontsize=12)
# plt.ylabel('Values', fontsize=12)
#
# # 增加图表边缘空白，使图形不那么拥挤
# plt.tight_layout()
#
# # 保存和显示图表
# plt.savefig('plots/actual_vs_predicted_boxplot.png')
# plt.show()