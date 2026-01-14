import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib.font_manager import FontProperties
import ast

# 设置matplotlib参数，适合SCI发表
plt.rcParams.update({
    'font.size': 9,           # 基础字体大小（调整为8）
    'axes.titlesize': 10,      # 子图标题字体大小（调整为9）
    'axes.labelsize': 10,      # 轴标签字体大小（调整为8）
    'xtick.labelsize': 9,     # x轴刻度标签字体大小（调整为7）
    'ytick.labelsize': 9,     # y轴刻度标签字体大小（调整为7）
    'legend.fontsize': 8,     # 图例默认字体大小（调整为6）
    'figure.titlesize': 11,    # 图形总标题字体大小
    'lines.linewidth': 1.3,   # 线条宽度
    'axes.linewidth': 0.9,    # 轴线宽度
    'grid.linewidth': 0.2,    # 网格线宽度
    'savefig.dpi': 300,       # 保存图片DPI
    'savefig.bbox': 'tight',  # 保存时自动调整边界
    'savefig.pad_inches': 0.1 # 保存时边距
})

# 设置seaborn样式
sns.set_style("whitegrid", {
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8
})

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlp_tuning.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureSelector(BaseEstimator, TransformerMixin):
    """选择指定特征列"""
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.selected_indices = None
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.selected_indices = [i for i, col in enumerate(X.columns) if col in self.feature_names]
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices].values
        return X[:, self.selected_indices]

def load_and_prepare_data(file_path):
    """加载并准备数据"""
    try:
        # 读取数据
        df = pd.read_excel(file_path)
        logger.info(f"数据加载成功: {df.shape[0]}行, {df.shape[1]}列")
        
        # 识别目标列和特征列
        target_col = 'log_EC50'
        possible_target_names = ['log_EC50', 'log EC50', 'logEC50', 'EC50_log', 'log10(EC50)']
        for name in possible_target_names:
            if name in df.columns:
                target_col = name
                break
        
        # 特征列 - 排除非特征列
        non_feature_cols = ['Name', 'Empirical formula', 'Canonical SMILES', 'log_EC50', 
                           'log EC50', 'logEC50', 'EC50_log', 'log10(EC50)']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        if not feature_cols:
            raise ValueError("未找到特征列！请检查数据格式")
        
        # 分离特征和目标
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"目标变量: {target_col}")
        logger.info(f"特征数量: {len(feature_cols)}")
        logger.info(f"特征示例: {feature_cols[:5]}...")
        
        # 处理缺失值 - 用中位数填充
        X = X.fillna(X.median())
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_cols, df
    
    except Exception as e:
        logger.error(f"数据准备错误: {str(e)}")
        raise

def calculate_mre(y_true, y_pred):
    """计算平均相对误差 (Mean Relative Error)"""
    # 避免除零错误
    y_true_nonzero = np.where(y_true != 0, y_true, 1e-10)
    relative_errors = np.abs(y_pred - y_true) / np.abs(y_true_nonzero)
    return np.mean(relative_errors)

def evaluate_model(y_true, y_pred):
    """评估模型性能 - 增加MRE指标"""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MRE': calculate_mre(y_true, y_pred)
    }
    return metrics

def train_mlp(X_train, X_test, y_train, y_test, feature_cols):
    """训练MLP模型（使用您提供的最佳参数）"""
    # 使用您提供的最佳参数
    best_params = {
        'activation': 'relu',
        'alpha': 0.001,
        'batch_size': 64,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'early_stopping': False,
        'epsilon': 1e-8,
        'hidden_layer_sizes': (100, 100),
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'max_fun': 15000,
        'max_iter': 1000,
        'momentum': 0.9,
        'n_iter_no_change': 10,
        'nesterovs_momentum': True,
        'power_t': 0.5,
        'random_state': 42,
        'shuffle': True,
        'solver': 'adam',
        'tol': 0.0001,
        'validation_fraction': 0.1,
        'verbose': False,
        'warm_start': False
    }
    
    # 创建模型的Pipeline
    mlp_pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(**best_params))
    ])
    
    logger.info("开始训练MLP模型...")
    start_time = time.time()
    
    # 训练模型
    mlp_pipeline.fit(X_train, y_train)
    
    logger.info(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    logger.info(f"使用参数: {best_params}")
    
    # 训练集和测试集预测
    y_train_pred = mlp_pipeline.predict(X_train)
    y_test_pred = mlp_pipeline.predict(X_test)
    
    # 评估模型
    train_metrics = evaluate_model(y_train, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    # 记录参数
    full_params = mlp_pipeline.named_steps['mlp'].get_params()
    
    # 保存结果
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'params': best_params,
        'full_params': full_params
    }
    
    return results, mlp_pipeline

def plot_scatter(X_train, X_test, y_train, y_test, feature_cols, results):
    """绘制MLP模型的散点图"""
    os.makedirs('mlp_results', exist_ok=True)
    
    # 创建模型Pipeline
    pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(**results['full_params']))
    ])
    pipeline.fit(X_train, y_train)
    
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # 定义颜色
    train_color = '#FF0000'  # 红色
    test_color = '#0000FF'   # 蓝色
    line_color = '#45B7D1'   # 青色
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(y_train, y_train_pred, alpha=0.8, s=25, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test, y_test_pred, alpha=0.8, s=25, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    # 绘制理想预测线
    all_y = np.concatenate([y_train, y_test])
    all_y_pred = np.concatenate([y_train_pred, y_test_pred])
    min_val = min(all_y.min(), all_y_pred.min())
    max_val = max(all_y.max(), all_y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    # 修改坐标轴标签 - 加粗字体
    ax.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax.set_title('MLP', fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴刻度标签加粗
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 性能标签
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    ax.text(0.95, 0.05, f'Training R² = {train_r2:.3f}\nTest R² = {test_r2:.3f}', 
             transform=ax.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8, frameon=True, 
              borderpad=0.15, labelspacing=0.1, handlelength=0.9, handletextpad=0.25)
    
    plt.tight_layout()
    plt.savefig('mlp_results/mlp_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_results/mlp_scatter.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("散点图已保存至 mlp_results/ 目录")

def save_scatter_data(X_train, X_test, y_train, y_test, feature_cols, results):
    """保存散点图数据到Excel文件"""
    os.makedirs('mlp_results', exist_ok=True)
    
    # 创建模型Pipeline
    pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(**results['full_params']))
    ])
    pipeline.fit(X_train, y_train)
    
    # 模型预测
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # 创建DataFrame保存数据
    scatter_data = pd.DataFrame({
        'Experimental_Train': y_train.values,
        'Predicted_Train': y_train_pred,
        'Set_Type': 'Training',
        'Index': y_train.index
    })
    
    test_data = pd.DataFrame({
        'Experimental_Test': y_test.values,
        'Predicted_Test': y_test_pred,
        'Set_Type': 'Test',
        'Index': y_test.index
    })
    
    # 合并训练和测试数据
    full_data = pd.concat([
        scatter_data.rename(columns={
            'Experimental_Train': 'Experimental',
            'Predicted_Train': 'Predicted'
        }),
        test_data.rename(columns={
            'Experimental_Test': 'Experimental',
            'Predicted_Test': 'Predicted'
        })
    ], ignore_index=True)
    
    # 保存到Excel
    with pd.ExcelWriter('mlp_results/scatter_plot_data.xlsx') as writer:
        full_data.to_excel(writer, sheet_name='All_Data', index=False)
        scatter_data.to_excel(writer, sheet_name='Training_Set', index=False)
        test_data.to_excel(writer, sheet_name='Test_Set', index=False)
        
        # 保存性能指标
        metrics_data = {
            'Metric': ['MSE', 'RMSE', 'MAE', 'R2', 'MRE'],
            'Train': [results['train_metrics'][m] for m in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']],
            'Test': [results['test_metrics'][m] for m in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']]
        }
        pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Performance_Metrics', index=False)
    
    logger.info("散点图数据已保存至 mlp_results/scatter_plot_data.xlsx")

def _calculate_average_feature_distance(train_features, other_features=None):
    """计算平均特征距离"""
    if other_features is None:
        distance_matrix = pairwise_distances(train_features, metric='euclidean')
        np.fill_diagonal(distance_matrix, np.nan)
        return np.nanmean(distance_matrix, axis=1)
    distance_matrix = pairwise_distances(other_features, train_features, metric='euclidean')
    return distance_matrix.mean(axis=1)

def plot_applicability_domain(model, X_train, X_test, y_train, y_test):
    """绘制调优模型适用域分析图并保存数据 - 修改为与参考代码一致的风格"""
    os.makedirs('mlp_results', exist_ok=True)

    feature_selector = model.named_steps['feature_selector']
    scaler = model.named_steps['scaler']

    X_train_selected = feature_selector.transform(X_train)
    X_test_selected = feature_selector.transform(X_test)

    X_train_scaled = scaler.transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    avg_dist_train = _calculate_average_feature_distance(X_train_scaled)
    avg_dist_test = _calculate_average_feature_distance(X_train_scaled, X_test_scaled)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    residuals_train = y_train_pred - y_train.to_numpy()
    residuals_test = y_test_pred - y_test.to_numpy()

    residual_std = residuals_train.std(ddof=1)
    if residual_std == 0:
        residual_std = 1e-10

    standardized_residual_train = residuals_train / residual_std
    standardized_residual_test = residuals_test / residual_std

    ad_threshold = avg_dist_train.mean() + 3 * avg_dist_train.std(ddof=1)
    residual_threshold = 3

    ad_df = pd.DataFrame({
        'Set': ['Training'] * len(avg_dist_train) + ['Testing'] * len(avg_dist_test),
        'AverageDistance': np.concatenate([avg_dist_train, avg_dist_test]),
        'StandardizedResidual': np.concatenate([standardized_residual_train, standardized_residual_test])
    })
    ad_df['OutOfDomain'] = (
        (np.abs(ad_df['StandardizedResidual']) > residual_threshold) | (ad_df['AverageDistance'] > ad_threshold)
    )

    # 修改为与参考代码一致的图形大小
    fig, ax = plt.subplots(figsize=(6, 5))

    # 使用参考代码中的颜色和标记样式
    for set_name, color, marker in [('Training', '#1f77b4', 'o'), ('Testing', '#ff7f0e', 's')]:
        subset = ad_df[ad_df['Set'] == set_name]
        ax.scatter(
            subset['AverageDistance'],
            subset['StandardizedResidual'],
            label=f"{set_name} Samples",
            alpha=0.85,
            s=18,
            color=color,
            marker=marker,
            edgecolors='white',
            linewidth=0.4
        )

    outliers = ad_df[ad_df['OutOfDomain']]
    if not outliers.empty:
        ax.scatter(
            outliers['AverageDistance'],
            outliers['StandardizedResidual'],
            facecolors='none',
            edgecolors='#d62728',
            linewidth=0.9,
            s=40,
            label='Outside Domain'
        )

    # 使用参考代码中的阈值线样式
    ax.axhline(residual_threshold, color='#d62728', linestyle='--', linewidth=0.9)
    ax.axhline(-residual_threshold, color='#d62728', linestyle='--', linewidth=0.9)
    ax.axvline(ad_threshold, color='#d62728', linestyle='--', linewidth=0.9)

    # 轴标签、标题、刻度文字设置与参考代码一致
    ax.set_xlabel('Average Feature Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standardized Residuals', fontsize=12, fontweight='bold')
    ax.set_title('Applicability Domain Analysis (Training & Testing)', fontsize=12, fontweight='bold')

    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')

    # 图例样式与参考代码一致
    ax.legend(
        fontsize=12,
        loc='upper right',
        framealpha=0.9,
        fancybox=True,
        shadow=False,
        borderpad=0.5,
        labelspacing=0.4,
        handlelength=0.1,
        handletextpad=0.6,
        columnspacing=0.5,
        prop={'weight': 'bold', 'size': 11}
    )
    ax.grid(True, linestyle='--', alpha=0.3)

    # 统计文本样式与参考代码一致
    exceed_text = (
        f"Training outside: {((ad_df['Set'] == 'Training') & ad_df['OutOfDomain']).sum()}\n"
        f"Testing outside: {((ad_df['Set'] == 'Testing') & ad_df['OutOfDomain']).sum()}"
    )
    ax.text(
        0.013,
        0.983,
        exceed_text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight='bold',
        fontfamily='Arial',
        color='darkred',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, 
                 edgecolor='orange', linewidth=0.9)
    )

    plt.tight_layout()
    # 保存为PNG和TIFF格式，与参考代码一致
    plt.savefig('mlp_results/mlp_applicability_domain.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_results/mlp_applicability_domain.tif', dpi=300, bbox_inches='tight', format='tiff')
    plt.close()

    testing_outliers = ((ad_df['Set'] == 'Testing') & ad_df['OutOfDomain']).sum()
    summary_df = pd.DataFrame({
        'Metric': ['Testing Samples Outside Domain'],
        'Count': [testing_outliers]
    })

    with pd.ExcelWriter('mlp_results/mlp_applicability_domain_data.xlsx') as writer:
        ad_df.to_excel(writer, sheet_name='All_Samples', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    logger.info("MLP适用域分析已保存至 mlp_results/ 目录")

def plot_results(results, X_train, X_test, y_train, y_test, feature_cols):
    """绘制其他结果图表"""
    os.makedirs('mlp_results', exist_ok=True)
    
    # 1. 模型性能指标
    metrics_df = pd.DataFrame({
        'Training': results['train_metrics'],
        'Test': results['test_metrics']
    }).T
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    colors = ['skyblue', 'salmon']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        metrics_df[metric].plot(kind='bar', ax=ax, color=colors)
        ax.set_title(f'{metric} Comparison', fontsize=7)
        ax.set_ylabel(metric, fontsize=6)
        ax.tick_params(axis='x', rotation=45, labelsize=5)
        
        # 添加数值标签
        for j, v in enumerate(metrics_df[metric]):
            ax.text(j, v, f'{v:.4f}', ha='center', va='bottom', fontsize=5)
    
    plt.tight_layout()
    plt.savefig('mlp_results/model_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_results/model_performance.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. 训练损失曲线
    mlp_model = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(**results['full_params']))
    ])
    mlp_model.fit(X_train, y_train)
    
    # 获取损失曲线
    loss_curve = mlp_model.named_steps['mlp'].loss_curve_
    
    plt.figure(figsize=(4, 3))
    plt.plot(loss_curve)
    plt.title('Training Loss Curve', fontsize=8)
    plt.xlabel('Iterations', fontsize=7)
    plt.ylabel('Loss', fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('mlp_results/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_results/loss_curve.pdf', bbox_inches='tight')
    plt.close()

def plot_residual_histogram(residuals, output_prefix, output_dir):
    """绘制残差分布直方图并保存为PNG和TIFF（图例缩小至图片1/4宽度）"""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(4, 3.5))
    sns.histplot(residuals, bins=30, stat='density', color='#4C72B0', edgecolor='white', alpha=0.8, label='Error Distribution')
    sns.kdeplot(residuals, color='#D62728', linewidth=2.0, label='Kernel Density')
    plt.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Zero Error')
    plt.title('Error Distribution of Testing Predictions', fontsize=12, fontweight='bold')
    plt.xlabel('Prediction Error (True - Predicted)', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(False)
    
    legend_font = FontProperties(
        size=9,        # 保持原字体大小
        weight='bold'  # 加粗
    )
    # 关键调整：缩小图例至约1/4宽度
    plt.legend(
        loc='upper right',
        prop=legend_font,  # 应用加粗字体
        frameon=True,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=0.6
    )
    
    plt.tight_layout()
    png_path = os.path.join(output_dir, f'{output_prefix}_residual_hist.png')
    tif_path = os.path.join(output_dir, f'{output_prefix}_residual_hist.tif')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(tif_path, dpi=300, bbox_inches='tight', format='tiff')
    plt.close()
    logger.info(f"残差直方图已保存至 {png_path} / {tif_path}")

def save_results_to_excel(results, feature_cols):
    """保存结果到Excel文件"""
    os.makedirs('mlp_results', exist_ok=True)
    
    # 创建Excel写入器
    with pd.ExcelWriter('mlp_results/mlp_results.xlsx') as writer:
        # 1. 模型性能比较
        performance_data = []
        for metric in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']:
            performance_data.append({
                'Metric': metric,
                'Training': results['train_metrics'][metric],
                'Test': results['test_metrics'][metric]
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_excel(writer, sheet_name='Model Performance', index=False)
        
        # 2. 训练集详细指标
        train_metrics_data = []
        for metric in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']:
            train_metrics_data.append({
                'Metric': metric,
                'Value': results['train_metrics'][metric]
            })
        
        train_metrics_df = pd.DataFrame(train_metrics_data)
        train_metrics_df.to_excel(writer, sheet_name='Training Set Metrics', index=False)
        
        # 3. 测试集详细指标
        test_metrics_data = []
        for metric in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']:
            test_metrics_data.append({
                'Metric': metric,
                'Value': results['test_metrics'][metric]
            })
        
        test_metrics_df = pd.DataFrame(test_metrics_data)
        test_metrics_df.to_excel(writer, sheet_name='Test Set Metrics', index=False)
        
        # 4. 模型参数
        params_df = pd.DataFrame([results['params']])
        params_df.to_excel(writer, sheet_name='Model Parameters', index=False)
        
        full_params_df = pd.DataFrame([results['full_params']])
        full_params_df.to_excel(writer, sheet_name='Full Parameters', index=False)
        
        # 5. 性能对比
        comparison_data = []
        for metric in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']:
            comparison_data.append({
                'Metric': metric,
                'Training': results['train_metrics'][metric],
                'Test': results['test_metrics'][metric],
                'Difference': results['train_metrics'][metric] - results['test_metrics'][metric] if metric != 'R2' else results['test_metrics'][metric] - results['train_metrics'][metric]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Training vs Test Comparison', index=False)

if __name__ == "__main__":
    try:
        input_file = "CaCo-2_molecular_descriptors_reduced.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info(f"开始处理文件: {input_file}")
        start_time = time.time()
        
        # 加载和准备数据
        X_train, X_test, y_train, y_test, feature_cols, df = load_and_prepare_data(input_file)
        
        # 训练模型（使用您提供的最佳参数）
        results, mlp_model = train_mlp(X_train, X_test, y_train, y_test, feature_cols)

        # 绘制残差直方图（测试集）
        test_residuals = (y_test - mlp_model.predict(X_test)).to_numpy()
        plot_residual_histogram(test_residuals, 'mlp_test', 'mlp_results')
        
        # 绘制散点图
        plot_scatter(X_train, X_test, y_train, y_test, feature_cols, results)
        
        # 保存散点图数据到Excel 
        save_scatter_data(X_train, X_test, y_train, y_test, feature_cols, results)

        # 绘制适用域分析
        plot_applicability_domain(mlp_model, X_train, X_test, y_train, y_test)
        
        # 绘制其他结果
        plot_results(results, X_train, X_test, y_train, y_test, feature_cols)
        
        # 保存结果到Excel
        save_results_to_excel(results, feature_cols)
        
        # 保存模型
        joblib.dump(mlp_model, 'mlp_results/mlp_model.pkl')
        
        # 记录执行时间
        elapsed = time.time() - start_time
        logger.info(f"\n总执行时间: {elapsed:.2f}秒")
        logger.info("MLP模型训练完成!")
        
        # 打印最终结果摘要
        print("\n" + "="*80)
        print("MLP TRAINING RESULTS SUMMARY")
        print("="*80)
        print("Model Performance:")
        print(f"  Training Set - MSE: {results['train_metrics']['MSE']:.4f}, RMSE: {results['train_metrics']['RMSE']:.4f}, MAE: {results['train_metrics']['MAE']:.4f}, R²: {results['train_metrics']['R2']:.4f}, MRE: {results['train_metrics']['MRE']:.4f}")
        print(f"  Test Set - MSE: {results['test_metrics']['MSE']:.4f}, RMSE: {results['test_metrics']['RMSE']:.4f}, MAE: {results['test_metrics']['MAE']:.4f}, R²: {results['test_metrics']['R2']:.4f}, MRE: {results['test_metrics']['MRE']:.4f}")
        print("\nModel Parameters:")
        for key, value in results['params'].items():
            print(f"  {key}: {value}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise