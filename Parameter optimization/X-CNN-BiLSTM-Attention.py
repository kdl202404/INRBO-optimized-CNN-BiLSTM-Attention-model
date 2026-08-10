import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Data Loading and Preprocessing
dataset = pd.read_excel(r"E:\Pytorch\LiKangdi\缝合\数据\数据\Vx 2.xlsx",
                        sheet_name="输出参数-总表",
                        skiprows=1)
print(dataset)

values = dataset.values[:, 1:]
values = np.array(values)

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

X_train = scaler_in.fit_transform(X_train)
X_test = scaler_in.transform(X_test)
X_val = scaler_in.transform(X_val)

Y_train = scaler_out.fit_transform(Y_train)
Y_test = scaler_out.transform(Y_test)
Y_val = scaler_out.transform(Y_val)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)

# Reshape input data to 3D (samples, time_steps, features)
X_train = X_train.unsqueeze(1)  # (samples, time_steps=1, features)
X_test = X_test.unsqueeze(1)
X_val = X_val.unsqueeze(1)


# 评价指标
import os
import random as rn
def set_my_seed(seed=123):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    np.random.seed(seed)                      # 设置 NumPy 随机数种子
    rn.seed(seed)                             # 设置 Python 内置随机数生成器种子
    torch.manual_seed(seed)                   # 设置 PyTorch 随机数种子
    torch.cuda.manual_seed(seed)              # 设置 GPU 随机数种子（如果使用 CUDA）
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 的结果可重复
    torch.backends.cudnn.benchmark = False    # 禁用基准模式以确保结果可重复


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

def evaluation(y_test, y_predict):
    # 计算 MAE、RMSE、IA、TIC
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    ia = index_of_agreement(y_test, y_predict)
    tic = theils_inequality_coefficient(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    # r_2=r2_score(y_test, y_predict)
    return rmse, mae, ia,tic  # r_2



# 自定义模型的构建
class CNNBiLSTMAttention(nn.Module):
    def __init__(self, hidden_size, output_size,num_layers):
        super(CNNBiLSTMAttention, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=X_train.shape[1], out_channels=64, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                              bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.conv1d(x)  # Convolutional layer
        x = self.maxpool(x)  # Max pooling layer
        x = x.view(x.size(0), -1, 64)  # Reshape to match LSTM input size

        lstm_out, _ = self.bilstm(x)  # LSTM layer
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # Attention mechanism
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # Weighted sum
        output = self.fc(context_vector)  # Fully connected layer
        return output




# Build the model function
def build_model(mode='CNNBiLSTMAttention', hidden_size=128,output_size=1,num_layers=1):
    # if mode == 'MLP':
    #     model = MLP(input_size, hidden_dim)
    # elif mode == 'LSTM':
    #     model = LSTMModel(input_size, hidden_dim)
    # elif mode == 'GRU':
    #     model = GRUModel(input_size, hidden_dim)
    # elif mode == 'Attention-LSTM':
    #     model = AttentionLSTMModel(input_size, hidden_dim)
    # elif mode == 'SSA-LSTM':
    #     model = SSALSTMModel(input_size, hidden_dim)
    # else:
    #     raise ValueError("Unsupported mode: {}".format(mode))
   if mode=='CNNBiLSTMAttention':
       model = CNNBiLSTMAttention( hidden_size=hidden_size,output_size=output_size, num_layers=num_layers)
   else:
        raise ValueError("Unsupported mode: {}".format(mode))
   return model


# 自定义画损失图函数和预测对比函数
def plot_loss(hist,imfname=''):
    plt.subplots(1,4,figsize=(16,2))
    for i,key in enumerate(hist.history.keys()):
        n=int(str('14')+str(i+1))
        plt.subplot(n)
        plt.plot(hist.history[key], 'k', label=f'Training {key}')
        plt.title(f'{imfname} Training {key}')
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
    plt.tight_layout()
    plt.show()
def plot_fit(y_test, y_pred):
    plt.figure(figsize=(10,8))
    plt.plot(y_pred, label='Predicted value', linestyle='--', marker='o', color='blue', linewidth=2)
    plt.plot(y_test, label='Actual value', linestyle='-', marker='x', color=(223 / 255, 143 / 255, 120 / 255),
             linewidth=2)
    r2 = r2_score(y_test, y_pred)
    plt.title(f"The prediction result :  ", fontsize=12)
    plt.xlabel("Sample points", fontsize=12)
    plt.ylabel("Speed(m/s)", fontsize=12)

    # 关键步骤：调整坐标轴范围，在上方留出空间
    plt.ylim(bottom=70,  # 保持数据区域下界 min(Y_test.min(), predictions.min()) * 0.95
             top=300)  # 上方多留8%高度 max(Y_test.max(), predictions.max()) * 1.08
    # 将图例精准定位在方框内的左上方空白区域
    plt.legend(
        loc='lower left',  # 初始定位左下（通过bbox_to_anchor微调）
        bbox_to_anchor=(0.55, 0.94),  # (x坐标, y坐标)：0.95表示靠近顶部
        ncol=2,  # 分2列紧凑排列
        frameon=True,  # 保留边框
        framealpha=0.8,  # 背景透明度
        edgecolor='#666666',  # 边框颜色
        facecolor='#F5F5F5',  # 背景颜色
        fontsize=12
    )

    plt.tight_layout()
    plt.show()


#
df_eval_all = pd.DataFrame(columns=['RMSE', 'MAE', 'IA','TIC'])
df_preds_all = pd.DataFrame()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 定义 train_fuc 函数
# 自定义 train_fuc 函数
def train_fuc(mode='CNNBiLSTMAttention', batch_size=64, epochs=30, hidden_size=128, num_layers=1, verbose=0, show_fit=True):
    global X_train, Y_train, X_test, Y_test
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将数据转换为 PyTorch 张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(Y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(Y_test).to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 构建模型
    model = build_model(mode=mode, hidden_size=hidden_size, num_layers=num_layers).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()  # 清零梯度
            output = model(batch_X)  # 前向传播
            loss = criterion(output, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        if verbose > 0 and (epoch + 1) % verbose == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # 预测
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.cpu().numpy()  # 转换为 NumPy 数组
        y_test_numpy = y_test_tensor.cpu().numpy()  # 确保测试标签也是 NumPy 数组格式
        y_pred = scaler_out.inverse_transform(y_pred)  # 反归一化
        y_test_numpy = scaler_out.inverse_transform(y_test_numpy)  # 对真实值进行反归一化
        if show_fit:
            y_train_pred_tensor = model(X_train_tensor)
            y_train_pred = y_train_pred_tensor.cpu().numpy()  # 转换为 NumPy 数组
            y_train_numpy = y_train_tensor.cpu().numpy()  # 确保测试标签也是 NumPy 数组格式
            y_train_pred = scaler_out.inverse_transform(y_train_pred)  # 反归一化
            y_train_numpy = scaler_out.inverse_transform(y_train_numpy)  # 对真实值进行反归一化


    # 可视化损失和拟合结果
    if show_fit:
        from plots import plot_all
        import numpy as np
        import pandas as pd
        r2 = r2_score(y_test_numpy, y_pred)
        print(f'\nR2:{r2}')
        plot_fit(y_test_numpy, y_pred)
        rmse, mae, ia, tic = evaluation(y_test_numpy, y_pred)
        # 输出评估结果
        table = PrettyTable(['测试集指标', 'RMSE', 'MAE', 'IA', 'TIC', 'R2'])
        table.add_row(['预测结果指标：', rmse, mae, ia, tic, f'{r2 * 100}%'])
        print(table)
        metrics = {
            'Metric': ['R²', 'MAE', 'RMSE', 'IA', 'TIC'],
            'Value': [r2, mae, rmse, ia, tic]
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


        # 将 Ytest 和 predicted_data 转换为浮点数类型
        Y_test = np.array(y_test_numpy, dtype=np.float64)
        predictions = np.array(y_pred, dtype=np.float64)
        Y_train = np.array(y_train_numpy, dtype=np.float64)
        predictions_train = np.array(y_train_pred, dtype=np.float64)
        # 检查是否有 NaN 或无穷大
        print(np.isnan(Y_test).sum(), np.isnan(predictions).sum())  # 打印 NaN 的数量
        print(np.isinf(Y_test).sum(), np.isinf(predictions).sum())  # 打印无穷大的数量
        print(np.isnan(Y_train).sum(), np.isnan(predictions_train).sum())  # 打印 NaN 的数量
        print(np.isinf(Y_train).sum(), np.isinf(predictions_train).sum())  # 打印无穷大的数量
        # 替换 NaN 或无穷大的值为 0（或其他合适的值）
        Y_test = np.nan_to_num(Y_test, nan=0.0, posinf=0.0, neginf=0.0)
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
        Y_train = np.nan_to_num(Y_train, nan=0.0, posinf=0.0, neginf=0.0)
        predictions_train = np.nan_to_num(predictions_train, nan=0.0, posinf=0.0, neginf=0.0)
        # 检查形状是否匹配
        print(Y_test.shape, predictions.shape)
        print(Y_train.shape, predictions_train.shape)
        # 如果不匹配，可以进行必要的调整，例如：
        Y_test = Y_test.reshape(predictions.shape)
        Y_train = Y_train.reshape(predictions_train.shape)
        plot_all(Y_train, predictions_train, Y_test, predictions)

        # 保存测试集的真实值和预测值到 Excel 文件
        result_df = pd.DataFrame({
            'Actual Value': Y_test.flatten(),
            'Predicted Value': predictions.flatten()
        })

        # 保存为 Excel 文件
        output_file = r"C:\Users\Administrator\Desktop\INRBO-CNN_LSTM_Attention预测结果.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"预测结果已保存到：{output_file}")


    # 评估模型
    eval_metrics = evaluation(y_test_numpy, y_pred)  # 确保评估输入的都是 NumPy 数组
    df_preds_all[mode] = y_pred.reshape(-1, )
    df_eval_all.loc[f'{mode}', :] = eval_metrics

    # 打印评估结果
    eval_metrics = [round(i, 3) for i in eval_metrics]
    r2 = r2_score(y_test_numpy, y_pred)
    print(f'{mode}的预测效果为：RMSE:{eval_metrics[0]}, MAE:{eval_metrics[1]}, IA:{eval_metrics[2]}, TIC:{eval_metrics[3]}')

    print("=======================================运行结束==========================================")

    return eval_metrics[0]



import matplotlib.pyplot as plt
import numpy as np
all_curve_data = []  # 用于存储多种算法的曲线数据

#
# from HEOA import HEOA
# def fobj(X):
#     s=train_fuc(mode='CNNBiLSTMAttention',batch_size=int(X[0]),epochs=int(X[1]),hidden_size=int(X[2]),num_layers=int(X[3]),verbose=0,show_fit=False)
#     return s
# GbestScore3,GbestPositon3,Curve1 = HEOA(pop=10,dim=4,lb=[16, 10, 64, 1] ,ub=[256, 1000, 128, 2],MaxIter=100,fun=fobj)
# GbestPositon3 = [int(i) for i in GbestPositon3.flatten()]
# # 假设 GbestPositon1 是一个标量或简单的可迭代对象
# print('HEOA最优适应度值：', GbestScore3)
# print(f"GbestPositon3: {GbestPositon3}, type: {type(GbestPositon3)}")
# if isinstance(GbestPositon3, np.ndarray) or isinstance(GbestPositon3, list):
#     GbestPositon3 = [int(i) for i in GbestPositon3]  # 对所有元素进行类型转换
# else:
#     GbestPositon3 = int(GbestPositon3)  # 如果是标量，直接转换
# print('HEOA最优解为：', GbestPositon3)
# # 保存 FTTA 算法的收敛曲线
# all_curve_data.append(("HEOA", Curve1))  # 保存算法名称和对应的收敛曲线
# print(type(Curve1))
# print(np.array(Curve1).shape)
#
# # 自己定义保存路径
# save_path = r'C:\Users\Administrator\Desktop\HEOA_Curve.csv'
# Curve1 = np.squeeze(Curve1)
# # 保存成csv
# curve_df = pd.DataFrame({'Iteration': np.arange(1, len(Curve1)+1), 'Fitness': Curve1})
# curve_df.to_csv(save_path, index=False, encoding='utf-8-sig')
# print(f"收敛曲线已保存到 {save_path}")
#
# train_fuc(mode='CNNBiLSTMAttention', batch_size=GbestPositon3[0], epochs=GbestPositon3[1], hidden_size=GbestPositon3[2], num_layers=GbestPositon3[3], verbose=1,show_fit=True)

#
# from SCNGO import SCNGO
# def fobj(X):
#     s=train_fuc(mode='CNNBiLSTMAttention',batch_size=int(X[0]),epochs=int(X[1]),hidden_size=int(X[2]),num_layers=int(X[3]),verbose=0,show_fit=False)
#     return s
# GbestScore4,GbestPositon4,Curve2 = SCNGO(Search_Agents=10,dimensions=4,Lowerbound=[16, 10, 64, 1] ,Upperbound=[256, 1000, 128, 2],Max_iterations=100,objective=fobj)
# # 假设 GbestPositon1 是一个标量或简单的可迭代对象
# print('SCNGO最优适应度值：', GbestScore4)
# print(f"GbestPositon4: {GbestPositon4}, type: {type(GbestPositon4)}")
# if isinstance(GbestPositon4, np.ndarray) or isinstance(GbestPositon4, list):
#     GbestPositon4 = [int(i) for i in GbestPositon4]  # 对所有元素进行类型转换
# else:
#     GbestPositon4 = int(GbestPositon4)  # 如果是标量，直接转换
# print('SCNGO最优解为：', GbestPositon4)
# # 保存 MPSO 算法的收敛曲线
# all_curve_data.append(("SCNGO", Curve2))  # 保存算法名称和对应的收敛曲线
#
# # 自己定义保存路径
# save_path = r'C:\Users\Administrator\Desktop\SCNGO_Curve.csv'
# # 保存成csv
# curve_df = pd.DataFrame({'Iteration': np.arange(1, len(Curve2)+1), 'Fitness': Curve2})
# curve_df.to_csv(save_path, index=False, encoding='utf-8-sig')
# print(f"收敛曲线已保存到 {save_path}")
#
# train_fuc(mode='CNNBiLSTMAttention', batch_size=GbestPositon4[0], epochs=GbestPositon4[1], hidden_size=GbestPositon4[2], num_layers=GbestPositon4[3], verbose=1,show_fit=True)

#
# from NRBO import NRBO
# def fobj(X):
#     s=train_fuc(mode='CNNBiLSTMAttention',batch_size=int(X[0]),epochs=int(X[1]),hidden_size=int(X[2]),num_layers=int(X[3]),verbose=0,show_fit=False)
#     return s
# GbestScore2,GbestPositon2,Curve3 = NRBO(N=10,dim=4,LB=[16, 10, 64, 1] ,UB=[256, 1000, 256, 4],MaxIt=100,fobj=fobj)
# # 假设 GbestPositon1 是一个标量或简单的可迭代对象
# print('NRBO最优适应度值：', GbestScore2)
# if isinstance(GbestPositon2, np.ndarray) or isinstance(GbestPositon2, list):
#     GbestPositon2 = [int(i) for i in GbestPositon2]  # 对所有元素进行类型转换
# else:
#     GbestPositon2 = int(GbestPositon2)  # 如果是标量，直接转换
# print('NRBO最优解为：', GbestPositon2)
# # 保存 NRBO 算法的收敛曲线
# all_curve_data.append(("NRBO", Curve3))  # 保存算法名称和对应的收敛曲线
#
# # 自己定义保存路径
# save_path = r'C:\Users\Administrator\Desktop\NRBO_Curve.csv'
# Curve3 = np.squeeze(Curve3)
# # 保存成csv
# curve_df = pd.DataFrame({'Iteration': np.arange(1, len(Curve3)+1), 'Fitness': Curve3})
# curve_df.to_csv(save_path, index=False, encoding='utf-8-sig')
# print(f"收敛曲线已保存到 {save_path}")
#
# train_fuc(mode='CNNBiLSTMAttention', batch_size=GbestPositon2[0], epochs=GbestPositon2[1], hidden_size=GbestPositon2[2], num_layers=GbestPositon2[3], verbose=1,show_fit=True)


from INRBO import NRBO
def fobj(X):
    s=train_fuc(mode='CNNBiLSTMAttention',batch_size=int(X[0]),epochs=int(X[1]),hidden_size=int(X[2]),num_layers=int(X[3]),verbose=0,show_fit=False)
    return s
GbestScore1,GbestPositon1,Curve4 = NRBO(N=50,dim=4,LB=[16, 10, 64, 1] ,UB=[256, 1000, 256, 4],MaxIt=100,fobj=fobj)
# 假设 GbestPositon1 是一个标量或简单的可迭代对象
print('INRBO最优适应度值：', GbestScore1)
if isinstance(GbestPositon1, np.ndarray) or isinstance(GbestPositon1, list):
    GbestPositon1 = [int(i) for i in GbestPositon1]  # 对所有元素进行类型转换
else:
    GbestPositon1 = int(GbestPositon1)  # 如果是标量，直接转换
print('INRBO最优解为：', GbestPositon1)
# 保存 INRBO 算法的收敛曲线
all_curve_data.append(("INRBO", Curve4))  # 保存算法名称和对应的收敛曲线

# 自己定义保存路径
save_path = r'C:\Users\Administrator\Desktop\INRBO_Curve.csv'
Curve4 = np.squeeze(Curve4)
# 保存成csv
curve_df = pd.DataFrame({'Iteration': np.arange(1, len(Curve4)+1), 'Fitness': Curve4})
curve_df.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"收敛曲线已保存到 {save_path}")

train_fuc(mode='CNNBiLSTMAttention', batch_size=GbestPositon1[0], epochs=GbestPositon1[1], hidden_size=GbestPositon1[2], num_layers=GbestPositon1[3], verbose=1,show_fit=True)



# # 绘制收敛曲线（迭代次数 vs 最优适应值）
# # plt.plot(range(len(Curve1)), Curve1)  # x轴是迭代次数，y轴是最佳适应度值
# # 绘制收敛曲线（多个算法的迭代次数 vs 最优适应值）
# plt.figure(figsize=(10, 6))  # 设置图形大小
# for algorithm, curve_data in all_curve_data:
#     plt.plot(range(len(curve_data)), curve_data, label=algorithm)  # 绘制每个算法的曲线
# plt.xlabel('The number of iterations', fontsize=12)
# plt.ylabel('RMSE', fontsize=12)
# plt.title('Convergence curve of algorithm', fontsize=12)
# plt.legend()  # 显示图例
# plt.show()


