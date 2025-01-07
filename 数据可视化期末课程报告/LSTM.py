import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 添加以下代码来支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=3, dropout=0.4):
        """
        LSTM模型定义 - 包含人口数据和生育态度数据
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 特征重要性权重
        self.feature_weights = nn.Parameter(torch.ones(input_size))
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 添加注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # 应用特征权重
        x = x * self.feature_weights.view(1, 1, -1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM层
        out, _ = self.lstm(x, (h0, c0))
        
        # 注意力机制
        attention_weights = self.attention(out)
        out = torch.sum(attention_weights * out, dim=1)
        
        # 全连接层
        out = self.fc(out)
        return out

class PopulationLSTM:
    def __init__(self, look_back=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化LSTM模型类
        
        Args:
            look_back (int): 用于预测的历史数据点数量
            device (str): 使用的设备（GPU/CPU）
        """
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.rate_scaler = MinMaxScaler(feature_range=(-1, 1))  # 用于缩放率数据
        self.device = device
        self.model = LSTMModel(input_size=8).to(device)  # 增加输入特征数
        
        # 添加特征权重
        self.attitude_weight = 2.0  # 生育态度权重
        self.rates_weight = 1.5    # 人口变化率权重
        self.trend_weight = 1.8    # 趋势权重
        
    def create_dataset(self, dataset, rates_data):
        """创建数据集，包含人口数据、人口变化率数据和生育态度数据"""
        X, y = [], []
        # 计算差分
        diff = np.diff(dataset.reshape(-1))
        
        # 生育态度数据
        negative_attitude = 0.455
        neutral_attitude = 0.297
        positive_attitude = 0.248
        
        # 缩放率数据
        scaled_rates = self.rate_scaler.fit_transform(rates_data)
        
        for i in range(len(dataset) - self.look_back - 1):
            features = []
            for j in range(self.look_back):
                if i + j < len(scaled_rates):
                    birth_rate = scaled_rates[i + j, 0]
                    death_rate = scaled_rates[i + j, 1]
                    natural_growth_rate = scaled_rates[i + j, 2]
                else:
                    birth_rate = scaled_rates[-1, 0]
                    death_rate = scaled_rates[-1, 1]
                    natural_growth_rate = scaled_rates[-1, 2]
                
                features.append([
                    dataset[i + j, 0],      # 原始人口值
                    diff[i + j],            # 人口差分值
                    birth_rate,             # 出生率
                    death_rate,             # 死亡率
                    natural_growth_rate,    # 自然增长率
                    negative_attitude,       # 负面态度
                    neutral_attitude,        # 中立态度
                    positive_attitude        # 正面态度
                ])
            X.append(features)
            y.append(dataset[i + self.look_back, 0])
        
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)
    
    def custom_loss(self, pred, target, alpha=0.7, beta=0.5):
        """自定义损失函数，人口变化趋势和生育态度"""
        # MSE损失
        mse_loss = nn.MSELoss()(pred, target)
        
        # 趋势损失
        if len(pred) > 1:
            pred_diff = pred[1:] - pred[:-1]
            target_diff = target[1:] - target[:-1]
            trend_loss = nn.MSELoss()(pred_diff, target_diff)
            
            # 添加额外的下降趋势惩罚
            if len(target_diff) >= 5:
                recent_trend = torch.mean(target_diff[-5:])
            else:
                recent_trend = torch.mean(target_diff)
            
            if len(pred_diff) > 0:
                # 增加趋势权重
                trend_direction_loss = torch.relu(pred_diff[-1] - recent_trend) * 2.0
                
                # 添加生育率和人口变化率的惩罚项
                birth_rate_penalty = torch.mean(torch.relu(pred_diff))  # 惩罚过快增长
                
                return (mse_loss + 
                        alpha * trend_loss * 1.5 +  # 趋势损失权重
                        beta * trend_direction_loss +  # 方向损失权重
                        0.5 * birth_rate_penalty)  # 添加生育率惩罚
            else:
                return mse_loss + alpha * trend_loss
        else:
            return mse_loss
    
    def train(self, data, rates_data, validation_split=0.2, epochs=300, batch_size=16):
        """训练模型"""
        # 数据预处理
        data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # 创建训练数据集
        X, y = self.create_dataset(data, rates_data)
        X = X.reshape(-1, self.look_back, 8)  # 8个特征
        
        # 确保有足够的数据进行验证
        min_train_size = 2  # 最小训练集大小
        if len(X) <= min_train_size:
            validation_split = 0
            print("警告：数据量太小，不进行验证集划分")
        
        # 分割训练集和验证集
        train_size = int(len(X) * (1 - validation_split))
        if validation_split > 0:
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y  # 使用全部数据作为验证集
        
        # 定义优化器和学习率调度器
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.003,  # 降低学习率
            weight_decay=0.02  # 增加权重衰减
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.3,  # 学习率调整
            patience=10,  # 减少耐心值
            verbose=True
        )
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = 40
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            train_pred = self.model(X_train)
            train_loss = self.custom_loss(train_pred, y_train.reshape(-1, 1))
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = self.custom_loss(val_pred, y_val.reshape(-1, 1))
            
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        return history
    
    def predict(self, data, rates_data, future_steps=5):
        """进行预测，加重生育态度的影响，保持平滑过渡"""
        self.model.eval()
        with torch.no_grad():
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
            scaled_rates = self.rate_scaler.transform(rates_data)
            
            # 计算最近的趋势
            recent_years = 3  # 考虑最近3年的趋势
            recent_diff = np.diff(data[-recent_years:])
            avg_annual_change = np.mean(recent_diff)  # 平均年度变化
            
            predictions = []
            current_batch = scaled_data[-self.look_back:]
            diff_data = np.diff(scaled_data.reshape(-1))
            
            # 生育态度数据
            negative_attitude = 0.455 * 0.6
            neutral_attitude = 0.297 * 0.2
            positive_attitude = 0.248 * 0.2
            
            # 计算初始下降率（基于最近趋势）
            initial_decline = avg_annual_change / data[-1]  # 相对变化率
            decline_rate = 1.0 + initial_decline  # 转换为乘数
            
            # 计算累积影响因子（从当前趋势平滑过渡到预期趋势）
            target_decline_rate = 0.996  # 目标年度下降率
            transition_steps = 3  # 过渡期（年）
            
            features = []
            for i in range(self.look_back):
                birth_rate = scaled_rates[-1, 0]
                death_rate = scaled_rates[-1, 1]
                natural_growth_rate = scaled_rates[-1, 2]
                
                if i < self.look_back - 1:
                    features.append([
                        float(current_batch[i][0]),
                        float(diff_data[-(self.look_back-1)+i]),
                        float(birth_rate) * self.rates_weight,
                        float(death_rate) * self.rates_weight,
                        float(natural_growth_rate) * self.rates_weight,
                        float(negative_attitude) * self.attitude_weight,
                        float(neutral_attitude) * self.attitude_weight,
                        float(positive_attitude) * self.attitude_weight
                    ])
                else:
                    features.append([
                        float(current_batch[-1][0]),
                        float(diff_data[-1]),
                        float(birth_rate) * self.rates_weight,
                        float(death_rate) * self.rates_weight,
                        float(natural_growth_rate) * self.rates_weight,
                        float(negative_attitude) * self.attitude_weight,
                        float(neutral_attitude) * self.attitude_weight,
                        float(positive_attitude) * self.attitude_weight
                    ])
            
            current_features = torch.FloatTensor(features).to(self.device)
            last_pred = float(data[-1])  # 上一年的实际值
            
            # 预测未来数据
            for step in range(future_steps):
                current_features_reshaped = current_features.reshape(1, self.look_back, 8)
                next_pred = self.model(current_features_reshaped)
                
                # 计算平滑过渡的下降率
                if step < transition_steps:
                    # 在过渡期内逐步从当前趋势过渡到目标趋势
                    transition_factor = step / transition_steps
                    current_decline_rate = decline_rate * (1 - transition_factor) + target_decline_rate * transition_factor
                else:
                    current_decline_rate = target_decline_rate
                
                # 应用平滑下降
                next_pred = torch.tensor(last_pred * current_decline_rate).to(self.device)
                
                # 根据生育态度调整预测值（减小即时影响）
                attitude_impact = (negative_attitude - positive_attitude) * 0.0005  # 减小影响因子
                next_pred = next_pred * (1 - attitude_impact * (step + 1))
                
                predictions.append(next_pred.cpu().numpy())
                last_pred = float(next_pred.item())  # 更新上一次的预测值
                
                # 更新特征，保持平滑过渡
                new_diff = float(next_pred.item() - current_features[-1, 0].item())
                new_feature = torch.tensor([[
                    float(next_pred.item()),
                    float(new_diff),
                    float(scaled_rates[-1, 0]) * (0.99 ** (step + 1)),  # 出生率
                    float(scaled_rates[-1, 1]) * (1.005 ** (step + 1)),  # 死亡率
                    float(scaled_rates[-1, 2]) * (0.98 ** (step + 1)),   # 自然增长率
                    float(negative_attitude) * (1.01 ** (step + 1)),     # 负面态度增强
                    float(neutral_attitude),
                    float(positive_attitude) * (0.99 ** (step + 1))      # 正面态度
                ]], dtype=torch.float32).to(self.device)
                
                current_features = torch.cat((current_features[1:], new_feature), 0)
            
            predictions = np.array(predictions).reshape(-1, 1)
            
            return predictions
    
    def plot_results(self, history, title='Model Training Loss'):
        """
        绘制训练结果
        
        Args:
            history: 训练历史
            title (str): 图表标题
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()  # 直接显示图表
    
    def predict_scenarios(self, data, rates_data, future_steps=5):
        """预测不同生育态度情景下的人口趋势"""
        scenarios = {
            '负面情景': {'negative': 0.455 * 0.6, 'neutral': 0.297 * 0.2, 'positive': 0.248 * 0.2},
            '中立情景': {'negative': 0.455 * 0.3, 'neutral': 0.297 * 0.5, 'positive': 0.248 * 0.2},
            '正面情景': {'negative': 0.455 * 0.2, 'neutral': 0.297 * 0.2, 'positive': 0.248 * 0.6}
        }
        
        predictions = {}
        for scenario, attitudes in scenarios.items():
            self.model.eval()
            with torch.no_grad():
                scaled_data = self.scaler.transform(data.reshape(-1, 1))
                scaled_rates = self.rate_scaler.transform(rates_data)
                
                # 计算最近的趋势
                recent_years = 3
                recent_diff = np.diff(data[-recent_years:])
                avg_annual_change = np.mean(recent_diff)
                
                current_batch = scaled_data[-self.look_back:]
                diff_data = np.diff(scaled_data.reshape(-1))
                
                # 使用不同情景的态度数据
                negative_attitude = attitudes['negative']
                neutral_attitude = attitudes['neutral']
                positive_attitude = attitudes['positive']
                
                # 根据情景调整下降率
                base_decline = avg_annual_change / data[-1]
                if scenario == '负面情景':
                    target_decline_rate = 0.996  # 每年下降0.4%
                elif scenario == '中立情景':
                    target_decline_rate = 0.998  # 每年下降0.2%
                else:  # 正面情景
                    target_decline_rate = 0.999  # 每年下降0.1%
                
                decline_rate = 1.0 + base_decline
                transition_steps = 3
                
                # 准备初始特征
                features = []
                for i in range(self.look_back):
                    birth_rate = scaled_rates[-1, 0]
                    death_rate = scaled_rates[-1, 1]
                    natural_growth_rate = scaled_rates[-1, 2]
                    
                    if i < self.look_back - 1:
                        features.append([
                            float(current_batch[i][0]),
                            float(diff_data[-(self.look_back-1)+i]),
                            float(birth_rate) * self.rates_weight,
                            float(death_rate) * self.rates_weight,
                            float(natural_growth_rate) * self.rates_weight,
                            float(negative_attitude) * self.attitude_weight,
                            float(neutral_attitude) * self.attitude_weight,
                            float(positive_attitude) * self.attitude_weight
                        ])
                    else:
                        features.append([
                            float(current_batch[-1][0]),
                            float(diff_data[-1]),
                            float(birth_rate) * self.rates_weight,
                            float(death_rate) * self.rates_weight,
                            float(natural_growth_rate) * self.rates_weight,
                            float(negative_attitude) * self.attitude_weight,
                            float(neutral_attitude) * self.attitude_weight,
                            float(positive_attitude) * self.attitude_weight
                        ])
                
                current_features = torch.FloatTensor(features).to(self.device)
                last_pred = float(data[-1])
                scenario_predictions = []
                
                # 预测未来数据
                for step in range(future_steps):
                    current_features_reshaped = current_features.reshape(1, self.look_back, 8)
                    next_pred = self.model(current_features_reshaped)
                    
                    # 计算平滑过渡的下降率
                    if step < transition_steps:
                        transition_factor = step / transition_steps
                        current_decline_rate = decline_rate * (1 - transition_factor) + target_decline_rate * transition_factor
                    else:
                        current_decline_rate = target_decline_rate
                    
                    # 应用平滑下降
                    next_pred = torch.tensor(last_pred * current_decline_rate).to(self.device)
                    
                    # 根据生育态度调整预测值
                    attitude_impact = (negative_attitude - positive_attitude) * 0.0005
                    next_pred = next_pred * (1 - attitude_impact * (step + 1))
                    
                    scenario_predictions.append(next_pred.cpu().numpy())
                    last_pred = float(next_pred.item())
                    
                    # 更新特征
                    new_diff = float(next_pred.item() - current_features[-1, 0].item())
                    new_feature = torch.tensor([[
                        float(next_pred.item()),
                        float(new_diff),
                        float(scaled_rates[-1, 0]) * (0.99 ** (step + 1)),
                        float(scaled_rates[-1, 1]) * (1.005 ** (step + 1)),
                        float(scaled_rates[-1, 2]) * (0.98 ** (step + 1)),
                        float(negative_attitude) * (1.01 ** (step + 1)),
                        float(neutral_attitude),
                        float(positive_attitude) * (0.99 ** (step + 1))
                    ]], dtype=torch.float32).to(self.device)
                    
                    current_features = torch.cat((current_features[1:], new_feature), 0)
                
                predictions[scenario] = np.array(scenario_predictions).reshape(-1, 1)
        
        return predictions
    
    def plot_population_pyramid(self, base_data, predictions, scenario='预测情景'):
        """
        绘制人口金字塔预测图
        """
        # 设置年龄组
        age_groups = list(range(0, 101, 5))  # 0-100岁，每5岁一组
        n_groups = len(age_groups) - 1
        
        # 估算年龄结构（基于当前人口金字塔形状，确保与年龄组数量匹配）
        age_distribution = np.array([
            4.8, 5.2, 5.8, 6.5, 7.2,  # 0-24岁
            8.0, 8.5, 8.8, 8.5, 8.0,  # 25-49岁
            7.5, 7.0, 6.5, 5.5, 4.5,  # 50-74岁
            3.0, 2.0, 1.0, 0.5, 0.2   # 75-100岁
        ]) / 100.0
        
        # 确保年龄分布总和为1
        age_distribution = age_distribution / np.sum(age_distribution)
        
        # 确保年龄分布数组长度与年龄组数量相匹配
        assert len(age_distribution) == n_groups, f"年龄分布数组长度({len(age_distribution)})与年龄组数量({n_groups})不匹配"
        
        # 性别比例
        male_ratio = 0.512
        female_ratio = 0.488
        
        plt.figure(figsize=(12, 8))
        
        # 计算每个年龄组的人口数量
        total_pop = predictions[-1][0]  # 使用最后一年的预测总人口
        male_pop = [total_pop * dist * male_ratio for dist in age_distribution]
        female_pop = [total_pop * dist * female_ratio for dist in age_distribution]
        
        # 年龄组标签
        y_labels = [f'{age_groups[i]}-{age_groups[i+1]}' for i in range(n_groups)]
        y_pos = np.arange(len(y_labels))
        
        # 绘制男性人口（左侧）
        plt.barh(y_pos, [-pop for pop in male_pop], height=0.8, 
                color='lightblue', alpha=0.7, label='男性')
        
        # 绘制女性人口（右侧）
        plt.barh(y_pos, female_pop, height=0.8,
                color='pink', alpha=0.7, label='女性')
        
        # 设置图表属性
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'2028年人口金字塔预测 - {scenario}')
        plt.ylabel('年龄组')
        plt.xlabel('人口（万人）')
        
        # 设置刻度
        max_pop = max(max(male_pop), max(female_pop))
        plt.xticks(np.arange(-max_pop, max_pop+1, max_pop/5),
                  [f'{abs(x):.0f}' for x in np.arange(-max_pop, max_pop+1, max_pop/5)])
        plt.yticks(y_pos, y_labels)
        
        # 添加图例
        plt.legend(loc='lower right')
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt

def main():
    """
    主函数，用于演示模型使用
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 2014-2023年的人口数据
    years = np.array(range(2014, 2024))
    population = np.array([
        137646, 138326, 139232, 140011, 140541,
        141008, 141212, 141260, 141175, 140967
    ], dtype=np.float32)  # 指定数据类型为float32
    
    # 添加人口变化率数据（2014-2023年）
    rates_data = np.array([
        [14.0, 7.2, 6.8],  # 2014年 [出生率, 死亡率, 自然增长率]
        [12.1, 7.1, 5.0],  # 2015年
        [13.0, 7.1, 5.9],  # 2016年
        [12.4, 7.1, 5.3],  # 2017年
        [10.9, 7.1, 3.8],  # 2018年
        [10.5, 7.2, 3.3],  # 2019年
        [8.5, 7.1, 1.4],   # 2020年
        [7.5, 7.2, 0.3],   # 2021年
        [6.8, 7.4, -0.6],  # 2022年
        [6.5, 7.5, -1.0]   # 2023年
    ], dtype=np.float32)  # 指定数据类型为float32
    
    # 创建LSTM模型实例
    model = PopulationLSTM(look_back=5)  # 减小look_back以适应较短的数据序列
    
    # 训练模型
    history = model.train(population, rates_data)
    
    # 绘制训练结果
    model.plot_results(history)
    
    # 预测不同情景下的人口趋势
    scenario_predictions = model.predict_scenarios(population, rates_data, future_steps=5)
    
    # 打印不同情景的预测结果
    for scenario, predictions in scenario_predictions.items():
        print(f"\n{scenario}下未来5年的人口预测（单位：万人）：")
        for i, pred in enumerate(predictions):
            print(f"{2024 + i}年: {pred[0]:.2f}")
    
    # 绘制不同情景的预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(years, population, label='历史数据', color='black')
    
    colors = {'负面情景': 'red', '中立情景': 'blue', '正面情景': 'green'}
    for scenario, predictions in scenario_predictions.items():
        plt.plot(range(2024, 2029), predictions, 
                label=f'{scenario}预测', 
                linestyle='--',
                color=colors[scenario])
    
    plt.title('不同生育态度情景下的人口趋势预测')
    plt.xlabel('年份')
    plt.ylabel('人口（万人）')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 绘制人口变化率趋势
    plt.figure(figsize=(12, 6))
    plt.plot(years, rates_data[:, 0], label='出生率(‰)', color='red')
    plt.plot(years, rates_data[:, 1], label='死亡率(‰)', color='blue')
    plt.plot(years, rates_data[:, 2], label='自然增长率(‰)', color='green')
    plt.title('人口变化率趋势')
    plt.xlabel('年份')
    plt.ylabel('比率(‰)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 为每个情景绘制人口金字塔
    for scenario, predictions in scenario_predictions.items():
        fig = model.plot_population_pyramid(population, predictions, scenario)
        fig.show()

if __name__ == "__main__":
    main()
