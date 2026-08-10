import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
rcParams['axes.unicode_minus'] = False    # 正确显示负号

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

X_train = filtered_values[n_train_number, :-1]
Y_train = filtered_values[n_train_number, -1].reshape(-1, 1)
X_test = filtered_values[n_test_number, :-1]
Y_test = filtered_values[n_test_number, -1].reshape(-1, 1)

# Split into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Normalize the features and labels
scaler_in = MinMaxScaler()
scaler_out = MinMaxScaler()

vp_X_train = scaler_in.fit_transform(X_train)
vp_X_test = scaler_in.transform(X_test)
vp_X_val = scaler_in.transform(X_val)

vt_Y_train = scaler_out.fit_transform(Y_train)
vt_Y_test = scaler_out.transform(Y_test)
vt_Y_val = scaler_out.transform(Y_val)


# Convert data to PyTorch tensors and reshape to (batch_size, channels, sequence_length)
X_train_tensor = torch.tensor(vp_X_train, dtype=torch.float32).permute(0, 1).unsqueeze(2)  # (batch_size, 6, 1)
Y_train_tensor = torch.tensor(vt_Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(vp_X_test, dtype=torch.float32).permute(0, 1).unsqueeze(2)
Y_test_tensor = torch.tensor(vt_Y_test, dtype=torch.float32)
X_val_tensor = torch.tensor(vp_X_val, dtype=torch.float32).permute(0, 1).unsqueeze(2)
Y_val_tensor = torch.tensor(vt_Y_val, dtype=torch.float32)






# Model Definition
class CNNBiLSTMAttention(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size):
        super(CNNBiLSTMAttention, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1)  # 6 input channels
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2  , 1)
        self.fc = nn.Linear(hidden_size * 2 , output_size)

    def forward(self, x):
        x = self.conv1d(x)  # Convolutional layer
        x = self.maxpool(x)  # Max pooling layer
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, sequence_length, features) for LSTM

        lstm_out, _ = self.bilstm(x)  # LSTM layer
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # Attention mechanism
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # Weighted sum
        output = self.fc(context_vector)  # Fully connected layer
        return output



# Model Initialization
hidden_size = 128
output_size = 1
batch_size=64
num_layers=1


model = CNNBiLSTMAttention( hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the Model
num_epochs = 100
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for i in range(0, X_train_tensor.size()[0], batch_size):
        batch_X = X_train_tensor[i:i + batch_size]
        batch_Y = Y_train_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Calculate validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, Y_val_tensor).item()

    # Store train and validation losses
    train_losses.append(epoch_loss / len(X_train_tensor))
    val_losses.append(val_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(X_train_tensor):.4f}, Val Loss: {val_loss:.4f}')

# Plot training and validation loss
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', linestyle='-', marker='o', markersize=2)
# plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', linestyle='-', marker='x', markersize=2)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.legend()
# plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions_train = model(X_train_tensor)
    predictions = scaler_out.inverse_transform(predictions.cpu().numpy())
    predictions_train = scaler_out.inverse_transform(predictions_train.cpu().numpy())

#
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

rmse, mae, ia,tic, r2 = evaluate_forecasts(Y_test, predictions)

# 输出评估结果
table = PrettyTable(['测试集指标', 'RMSE', 'MAE','IA','TIC', 'R2'])
table.add_row(['预测结果指标：', rmse, mae, ia,tic, f'{r2 * 100}%'])
print(table)
metrics = {
    'Metric': ['R²', 'MAE', 'RMSE','IA','TIC'],
    'Value': [r2, mae, rmse, ia,tic]
}

df_metrics = pd.DataFrame(metrics)
# plt.figure(figsize=(6, 2))
# # plt.title("CNN-BiLSTM-Attention")
# plt.axis('tight')
# plt.axis('off')
# table = plt.table(cellText=df_metrics.values, colLabels=df_metrics.columns, cellLoc='center', loc='center')
# table.scale(1, 2)  # 调整表格的大小
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# plt.show()



# Visualization
# plt.figure(figsize=(8, 6))
# plt.plot(predictions, label='Predicted value', linestyle='--', marker='o', color='blue', linewidth=2)
# plt.plot(Y_test, label='Actual value', linestyle='-', marker='x', color=(223/255, 143/255, 120/255), linewidth=2)
# plt.xlabel("Sample points",fontsize=12)
# plt.ylabel("Speed(m/s)",fontsize=12)
# plt.title(f"The prediction result : ",fontsize=12)

plt.figure(figsize=(8, 6))
plt.plot(predictions, label='预测值', linestyle='--', marker='o', color='blue', linewidth=2)
plt.plot(Y_test, label='实际值', linestyle='-', marker='x', color=(223/255, 143/255, 120/255), linewidth=2)
plt.xlabel("实验点",fontsize=12)
plt.ylabel("速度(m/s)",fontsize=12)
plt.title(f"预测结果: ")
# 坐标刻度字体大小和颜色
plt.xticks(fontsize=13, color='black')
plt.yticks(fontsize=13, color='black')


# 关键步骤：调整坐标轴范围，在上方留出空间
plt.ylim(bottom=70,  # 保持数据区域下界 min(Y_test.min(), predictions.min()) * 0.95
         top=300)     # 上方多留8%高度 max(Y_test.max(), predictions.max()) * 1.08
# 将图例精准定位在方框内的左上方空白区域
plt.legend(
    loc='lower left',          # 初始定位左下（通过bbox_to_anchor微调）
    bbox_to_anchor=(0.63, 0.92),  # (x坐标, y坐标)：0.95表示靠近顶部
    ncol=2,                      # 分2列紧凑排列
    frameon=True,                # 保留边框
    framealpha=0.8,              # 背景透明度
    edgecolor='#666666',         # 边框颜色
    facecolor='#F5F5F5',         # 背景颜色
    fontsize=12
)
plt.tight_layout()
save_path = r"D:\Desktop\论文\图片格式\高清jpg-预测图\c-l.jpg"  # 修改为你想保存的路径
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 先保存

plt.show()

# # 保存测试集的真实值和预测值到 Excel 文件
# result_df = pd.DataFrame({
#     'Actual Value': Y_test.flatten(),
#     'Predicted Value': predictions.flatten()
# })
#
# # 保存为 Excel 文件
# output_file = r"D:\Desktop\CNN_BiLSTM-Attention预测结果.xlsx"
# result_df.to_excel(output_file, index=False)
# print(f"预测结果已保存到：{output_file}")


# from plots import plot_all
# import numpy as np
# import pandas as pd
# # 将 Ytest 和 predicted_data 转换为浮点数类型
# Y_test = np.array(Y_test, dtype=np.float64)
# predictions = np.array(predictions, dtype=np.float64)
# Y_train = np.array(Y_train, dtype=np.float64)
# predictions_train = np.array(predictions_train, dtype=np.float64)
# # 检查是否有 NaN 或无穷大
# print(np.isnan(Y_test).sum(), np.isnan(predictions).sum())  # 打印 NaN 的数量
# print(np.isinf(Y_test).sum(), np.isinf(predictions).sum())  # 打印无穷大的数量
# print(np.isnan(Y_train).sum(), np.isnan(predictions_train).sum())  # 打印 NaN 的数量
# print(np.isinf(Y_train).sum(), np.isinf(predictions_train).sum())  # 打印无穷大的数量
# # 替换 NaN 或无穷大的值为 0（或其他合适的值）
# Y_test = np.nan_to_num(Y_test, nan=0.0, posinf=0.0, neginf=0.0)
# predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
# Y_train = np.nan_to_num(Y_train, nan=0.0, posinf=0.0, neginf=0.0)
# predictions_train = np.nan_to_num(predictions_train, nan=0.0, posinf=0.0, neginf=0.0)
# # 检查形状是否匹配
# print(Y_test.shape, predictions.shape)
# print(Y_train.shape, predictions_train.shape)
# # 如果不匹配，可以进行必要的调整，例如：
# Y_test = Y_test.reshape(predictions.shape)
# Y_train = Y_train.reshape(predictions_train.shape)
# # path='D:\Desktop\缝合\程序\RMSE.xlsx'
# plot_all(Y_train,predictions_train,Y_test, predictions)





# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 示例数据（请替换为您的 y_test 和 y_test_predicted 数据）
# # 将 Ytest 和 predicted_data 转换为浮点数类型并进行一维化
# Ytest = np.array(Y_test, dtype=np.float64).flatten()
# predicted_data = np.array(predictions, dtype=np.float64).flatten()
#
# # 创建 DataFrame
# data = pd.DataFrame({
#     'Actual': Ytest,
#     'Predicted': predicted_data
# })
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


# import numpy as np
# import pandas as pd
# import torch

# # 预测函数
# def predict_with_lstm(model, X):
#     model.eval()
#     with torch.no_grad():
#         yhat = model(X)
#     return yhat
#
# # 对多组输入特征进行预测
# y_preds = predict_with_lstm(model, X_test_tensor)
# # 反归一化
# predicted_data = scaler_out.inverse_transform(y_preds.cpu().numpy())
#
# features = X_test  # 获取测试集的特征值
# # 创建DataFrame存储所有耗气量和对应特征值
# results_df = pd.DataFrame(features, columns=[f'特征{i}' for i in range(features.shape[1])])  # Ensure correct columns
# results_df['预测速度'] = predicted_data.flatten()  # Flatten if necessary
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
# results_df.to_excel("cnn_bilstm_attention_results.xlsx", index=False)