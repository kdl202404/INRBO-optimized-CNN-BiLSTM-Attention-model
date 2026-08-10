import os
import math
import epochs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# 读取 Excel 数据
dataset = pd.read_excel(r"D:\Desktop\缝合\喷嘴\数据\Vx 2.xlsx",
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

# 转换为张量
vp_train_tensor = torch.tensor(vp_train, dtype=torch.float32).reshape(-1,6,1)
vt_train_tensor = torch.tensor(vt_train, dtype=torch.float32)
vp_test_tensor = torch.tensor(vp_test, dtype=torch.float32).reshape(-1,6,1)
vt_test_tensor = torch.tensor(vt_test, dtype=torch.float32)

train_dataset = TensorDataset(vp_train_tensor, vt_train_tensor)
test_dataset = TensorDataset(vp_test_tensor, vt_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_size)
        attention_scores = self.attention_weights(x)  # shape: (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # shape: (batch_size, sequence_length, 1)
        weighted_sum = torch.sum(x * attention_weights, dim=1)  # shape: (batch_size, hidden_size)
        return weighted_sum

class CNNGRUAttention(nn.Module):
    def __init__(self):
        super(CNNGRUAttention, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1)
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.attention = Attention(128)  # Attention Layer
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1d(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1, 64)  # 重塑为 (batch_size, sequence_length, 64)
        x, _ = self.gru(x)  # GRU 层，输出形状为 (batch_size, sequence_length, 128)
        x = self.attention(x)  # Attention 层，输出形状为 (batch_size, 128)
        x = self.fc(x)  # 全连接层
        return x



model = CNNGRUAttention()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 创建一个空列表来存储每个 epoch 的训练和验证损失
train_loss_values = []
val_loss_values = []
num_epochs = 200
for epoch in range(num_epochs):
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
# # 输出表格
# print(df_metrics)
# 可视化表格输出
plt.figure(figsize=(6, 2))
# plt.title("CNN-GRU-Attention")
plt.axis('tight')
plt.axis('off')
table = plt.table(cellText=df_metrics.values, colLabels=df_metrics.columns, cellLoc='center', loc='center')
table.scale(1, 2)  # 调整表格的大小
table.auto_set_font_size(False)
table.set_fontsize(12)
plt.show()


# 绘制损失图
plt.figure(figsize=(8, 6))
plt.plot(train_loss_values, label='Training Loss', linestyle='-', marker='o', markersize=2)
plt.plot(val_loss_values, label='Validation Loss', linestyle='-', marker='x', markersize=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
# plt.grid()  # 添加网格以提高可读性
plt.show()


# import matplotlib.pyplot as plt
# from matplotlib import font_manager
#
# # 设置使用黑体字体
# font_path = "C:/Windows/Fonts/simhei.ttf"  # 根据需要替换为其他中文字体
# my_font = font_manager.FontProperties(fname=font_path)
#
# # 绘制结果图
# plt.figure(figsize=(8, 6))
# plt.plot(predicted_data, label='预测值', linestyle='--', marker='o', color='blue', linewidth=2)
# plt.plot(Ytest, label='真实值', linestyle='-', marker='x', color=(223/255, 143/255, 120/255), linewidth=2)
#
# # 设置标签、标题和图例的字体
# plt.xlabel("监测点", fontsize=14, fontproperties=my_font)
# plt.ylabel("速度（m/s）", fontsize=14, fontproperties=my_font)
# plt.title(f"模型预测结果 :\nR2: {r2} %", fontsize=14, fontproperties=my_font)
#
# # 设置图例的字体，并将'预测值'图例放置到标题右边
# plt.legend(fontsize=12, prop=my_font, loc='upper left', bbox_to_anchor=(0.85, 1.12))
#
# # 显示图形
# plt.show()



# 绘制结果图
plt.figure(figsize=(10, 8))
plt.plot(predicted_data, label='Predicted value', linestyle='--', marker='o', color='blue', linewidth=2)
plt.plot(Ytest, label='Actual value', linestyle='-', marker='x', color=(223/255, 143/255, 120/255), linewidth=2)
# 设置标签、标题和图例的字体
plt.xlabel("Sample points",fontsize=12)
plt.ylabel("Speed(m/s)",fontsize=12)
plt.title(f"The prediction result :",fontsize=12)

# 关键步骤：调整坐标轴范围，在上方留出空间
plt.ylim(bottom=70,  # 保持数据区域下界 min(Ytest.min(), predicted_data.min()) * 0.95
         top=300)     # 上方多留8%高度
# 将图例精准定位在方框内的左上方空白区域
plt.legend(
    loc='lower left',          # 初始定位左下（通过bbox_to_anchor微调）
    bbox_to_anchor=(0.55, 0.94),  # (x坐标, y坐标)：0.95表示靠近顶部
    ncol=2,                      # 分2列紧凑排列
    frameon=True,                # 保留边框
    framealpha=0.8,              # 背景透明度
    edgecolor='#666666',         # 边框颜色
    facecolor='#F5F5F5',         # 背景颜色
    fontsize=12
)

plt.tight_layout()
plt.show()

# # 保存测试集的真实值和预测值到 Excel 文件
# result_df = pd.DataFrame({
#     'Actual Value': Ytest.flatten(),
#     'Predicted Value': predicted_data.flatten()
# })
#
# # 保存为 Excel 文件
# output_file = r"D:\Desktop\CNNGRUAttention预测结果.xlsx"
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


#
# #箱线图
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# # 示例数据（请替换为您的 y_test 和 y_test_predicted 数据）
# # 将 Ytest 和 predicted_data 转换为浮点数类型并进行一维化
# Ytest = np.array(Ytest, dtype=np.float64).flatten()
# predicted_data = np.array(predicted_data, dtype=np.float64).flatten()
# # 创建 DataFrame
# data = pd.DataFrame({
#     'Actual': Ytest,
#     'Predicted': predicted_data
# })
# # 转换为长格式，便于绘图
# data_melted = data.melt(var_name='Type', value_name='Value')
# # 设置风格和调色板
# sns.set(style="whitegrid")
# # 设置图像大小和分辨率
# plt.figure(dpi=120)
# # 绘制箱线图
# sns.boxplot(x='Type', y='Value', data=data_melted, palette="deep")
# # 设置标题和轴标签
# plt.title('Boxplot of Actual and Predicted Values', fontsize=16, fontweight='bold')
# plt.xlabel('Value Type', fontsize=12)
# plt.ylabel('Values', fontsize=12)
# # 增加图表边缘空白，使图形不那么拥挤
# plt.tight_layout()
# # 保存和显示图表
# plt.savefig('plots/actual_vs_predicted_boxplot.png')
# plt.show()

# import numpy as np
# import pandas as pd
# import torch
#
# # 预测函数
# def predict_with_lstm(model, X):
#     model.eval()
#     with torch.no_grad():
#         yhat = model(X)
#     return yhat
#
# # 对多组输入特征进行预测
# y_preds = predict_with_lstm(model, vp_test_tensor)
# # 反归一化
# predicted_data = m_out.inverse_transform(y_preds.cpu().numpy())
#
# features = Xtest  # 获取测试集的特征值
# # 创建DataFrame存储所有耗气量和对应特征值
# results_df = pd.DataFrame()
# results_df[['特征' + str(i) for i in range(features.shape[1])]] = features
# results_df['预测速度'] = predicted_data
#
# # 查找最小耗气量及其对应特征值
# min_consumption = results_df['预测速度'].min()
# min_index = results_df['预测速度'].idxmin()
# min_features = results_df.iloc[min_index, :-1]  # 获取对应的特征值
#
# # 输出最小耗气量及其对应的特征值
# print(f"最小速度: {min_consumption}, 对应特征值: {min_features.values}")
#
# # 如果需要，将结果保存到Excel文件
# results_df.to_excel("cnn_gru_attention_results.xlsx", index=False)

