import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from prettytable import PrettyTable
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
rcParams['axes.unicode_minus'] = False    # 正确显示负号
# 读取数据
dataset = pd.read_excel(r"D:\Desktop\缝合\主喷嘴\数据\Vx 2.xlsx",
                        sheet_name="输出参数-总表",
                        skiprows=1)

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

vp_train = vp_train.reshape((vp_train.shape[0], vp_train.shape[1], 1))
vp_test = vp_test.reshape((vp_test.shape[0], vp_test.shape[1], 1))

# 转换为Tensor
vp_train_tensor = torch.tensor(vp_train, dtype=torch.float32)
vt_train_tensor = torch.tensor(vt_train, dtype=torch.float32)
vp_test_tensor = torch.tensor(vp_test, dtype=torch.float32)
vt_test_tensor = torch.tensor(vt_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(vp_train_tensor, vt_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 定义CNN-LSTM模型
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128,num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1d(x)  # (batch_size, 64, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, 64)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)
        return x


# 创建模型
model = CNNLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 创建一个空列表来存储每个 epoch 的训练和验证损失
train_loss_values = []
val_loss_values = []

# 训练模型
epochs = 300
for epoch in range(epochs):
    model.train()
    epoch_loss = 0  # 初始化当前 epoch 的损失
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  # 累加当前 batch 的损失

    avg_train_loss = epoch_loss / len(train_loader)  # 计算平均训练损失
    train_loss_values.append(avg_train_loss)  # 保存平均训练损失

    # 计算验证集损失
    model.eval()
    with torch.no_grad():
        val_outputs = model(vp_test_tensor)
        val_loss = criterion(val_outputs, vt_test_tensor)
        val_loss_values.append(val_loss.item())  # 保存验证损失

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')

# 预测
model.eval()
with torch.no_grad():
    yhat = model(vp_test_tensor)
    yhat_train = model(vp_train_tensor)
    # 反归一化
    predicted_data = m_out.inverse_transform(yhat.numpy())
    predicted_data_train = m_out.inverse_transform(yhat_train.numpy())

# # 计算评估指标
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
# # # 输出表格
# # print(df_metrics)
# # 可视化表格输出
# plt.figure(figsize=(6, 2))
# # plt.title("CNN-LSTM")
# plt.axis('tight')
# plt.axis('off')
# table = plt.table(cellText=df_metrics.values, colLabels=df_metrics.columns, cellLoc='center', loc='center')
# table.scale(1, 2)  # 调整表格的大小
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# plt.show()

# # 绘制损失图
# plt.figure(figsize=(8, 6))
# plt.plot(train_loss_values, label='Training Loss', linestyle='-', marker='o', markersize=2)
# plt.plot(val_loss_values, label='Validation Loss', linestyle='-', marker='x', markersize=2)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss Over Epochs")
# plt.legend()
# # plt.grid()  # 添加网格以提高可读性
# plt.show()


# # 绘制结果图
# plt.figure(figsize=(10, 8))
# plt.plot(predicted_data, label='Predicted value', linestyle='--', marker='o', color='blue', linewidth=2)
# plt.plot(Ytest, label='Actual value', linestyle='-', marker='x', color=(223/255, 143/255, 120/255), linewidth=2)
# plt.xlabel("Sample points",fontsize=12)
# plt.ylabel("Speed(m/s)",fontsize=12)
# plt.title(f"The prediction result : ",fontsize=12)
#
# # 关键步骤：调整坐标轴范围，在上方留出空间
# plt.ylim(bottom=70,  # 保持数据区域下界 min(Ytest.min(), predicted_data.min()) * 0.95
#          top=300)     # 上方多留8%高度 max(Ytest.max(), predicted_data.max()) * 1.08
# # 将图例精准定位在方框内的左上方空白区域
# plt.legend(
#     loc='lower left',          # 初始定位左下（通过bbox_to_anchor微调）
#     bbox_to_anchor=(0.55, 0.94),  # (x坐标, y坐标)：0.95表示靠近顶部
#     ncol=2,                      # 分2列紧凑排列
#     frameon=True,                # 保留边框
#     framealpha=0.8,              # 背景透明度
#     edgecolor='#666666',         # 边框颜色
#     facecolor='#F5F5F5',         # 背景颜色
#     fontsize=12
# )
#
# plt.tight_layout()
# plt.show()

# # 保存测试集的真实值和预测值到 Excel 文件
# result_df = pd.DataFrame({
#     'Actual Value': Ytest.flatten(),
#     'Predicted Value': predicted_data.flatten()
# })
#
# # 保存为 Excel 文件
# output_file = r"D:\Desktop\CNN_LSTM预测结果.xlsx"
# result_df.to_excel(output_file, index=False)
# print(f"预测结果已保存到：{output_file}")


from plots import plot_all
import numpy as np
import pandas as pd
# 将 Ytest 和 predicted_data 转换为浮点数类型
Ytest = np.array(Ytest, dtype=np.float64)
predicted_data = np.array(predicted_data, dtype=np.float64)
Ytrain = np.array(Ytrain, dtype=np.float64)
predicted_data_train = np.array(predicted_data_train, dtype=np.float64)
# 检查是否有 NaN 或无穷大
print(np.isnan(Ytest).sum(), np.isnan(predicted_data).sum())  # 打印 NaN 的数量
print(np.isinf(Ytest).sum(), np.isinf(predicted_data).sum())  # 打印无穷大的数量
print(np.isnan(Ytrain).sum(), np.isnan(predicted_data_train).sum())  # 打印 NaN 的数量
print(np.isinf(Ytrain).sum(), np.isinf(predicted_data_train).sum())  # 打印无穷大的数量
# 替换 NaN 或无穷大的值为 0（或其他合适的值）
Ytest = np.nan_to_num(Ytest, nan=0.0, posinf=0.0, neginf=0.0)
predicted_data = np.nan_to_num(predicted_data, nan=0.0, posinf=0.0, neginf=0.0)
Ytrain = np.nan_to_num(Ytrain, nan=0.0, posinf=0.0, neginf=0.0)
predicted_data_train = np.nan_to_num(predicted_data_train, nan=0.0, posinf=0.0, neginf=0.0)
# 检查形状是否匹配
print(Ytest.shape, predicted_data.shape)
print(Ytrain.shape, predicted_data_train.shape)
# 如果不匹配，可以进行必要的调整，例如：
Ytest = Ytest.reshape(predicted_data.shape)
Ytrain = Ytrain.reshape(predicted_data_train.shape)
plot_all(Ytrain,predicted_data_train,Ytest, predicted_data)
