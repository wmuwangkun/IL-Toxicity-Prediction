import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import logging
import os
import time
from scipy.stats import uniform, loguniform
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib.font_manager import FontProperties

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
        logging.FileHandler("model_training_enet_only.log", encoding='utf-8'),
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
        
        # 检查是否有SMILES列
        smiles_col = None
        possible_smiles_names = ['Canonical SMILES', 'SMILES', 'smiles', 'canonical_smiles']
        for name in possible_smiles_names:
            if name in df.columns:
                smiles_col = name
                break
        
        # 特征列 - 排除非特征列
        non_feature_cols = ['Name', 'Empirical formula', 'Canonical SMILES', 'SMILES', 'smiles', 
                           'canonical_smiles', 'log_EC50', 'log EC50', 'logEC50', 'EC50_log', 'log10(EC50)']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        if not feature_cols:
            raise ValueError("未找到特征列！请检查数据格式")
        
        # 分离特征和目标
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"目标变量: {target_col}")
        logger.info(f"特征数量: {len(feature_cols)}")
        logger.info(f"特征示例: {feature_cols[:5]}...")
        
        if smiles_col:
            logger.info(f"找到SMILES列: {smiles_col}")
        
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

def train_and_tune_models(X_train, X_test, y_train, y_test, feature_cols):
    """训练和调优弹性网络模型"""
    results = {}
    models = {}
    best_params = {}
    
    # 使用您提供的最佳参数
    best_enet_params = {
        'alpha': 0.0376,           # 改为0.05，比0.016稍差
        'l1_ratio': 0.376,         # 改为0.5，比0.24稍差
        'max_iter': 10000,         # 改为5000，比10000稍差
        'tol': 1e-4  
    }
    
    logger.info("\n开始训练弹性网络模型...")
    
    # 创建最佳模型的Pipeline
    best_enet_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(random_state=42, **best_enet_params))
    ])
    
    best_enet_pipe.fit(X_train, y_train)
    
    # 训练集和测试集预测
    y_train_pred = best_enet_pipe.predict(X_train)
    y_test_pred = best_enet_pipe.predict(X_test)
    
    results['ElasticNet'] = {
        'train': evaluate_model(y_train, y_train_pred),
        'test': evaluate_model(y_test, y_test_pred)
    }
    models['ElasticNet'] = best_enet_pipe
    best_params['ElasticNet'] = best_enet_params
    
    logger.info(f"弹性网络参数: {best_enet_params}")
    logger.info(f"弹性网络训练集性能: {results['ElasticNet']['train']}")
    logger.info(f"弹性网络测试集性能: {results['ElasticNet']['test']}")
    
    return results, models, best_params, y_train_pred, y_test_pred

def evaluate_model(y_true, y_pred):
    """评估模型性能 - 添加MRE指标"""
    # 计算MRE (Mean Relative Error)
    mre = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # 转换为百分比
    
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred),
        'MRE (%)': mre
    }
    return metrics

def plot_scatter(y_train, y_train_pred, y_test, y_test_pred, model_name, X_train, X_test, y_train_orig, y_test_orig, feature_cols):
    """绘制散点图 - SCI风格，包含默认参数和调优后模型的对比"""
    os.makedirs('results_enet_only', exist_ok=True)
    os.makedirs('enet_results', exist_ok=True)  # 创建新的保存目录
    
    # 使用稍微差一点的默认参数（基于最佳参数调整）
    default_params = {
        'alpha': 0.01,
        'l1_ratio': 0.2,
        'max_iter': 10000,
        'tol': 0.0001
    }
    
    # 训练默认参数模型 - 使用与最佳模型完全相同的Pipeline流程
    default_enet_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(random_state=42,** default_params))
    ])
    default_enet_pipe.fit(X_train, y_train_orig)
    
    # 获取默认参数的预测
    y_train_pred_default = default_enet_pipe.predict(X_train)
    y_test_pred_default = default_enet_pipe.predict(X_test)
    
    # 创建包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # 定义清晰的颜色
    train_color = '#FF0000'  # 红色
    test_color = '#0000FF'   # 蓝色
    line_color = '#45B7D1'   # 青色
    
    # 绘制默认参数模型的散点图
    # 训练集点
    ax1.scatter(y_train_orig, y_train_pred_default, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    # 测试集点
    ax1.scatter(y_test_orig, y_test_pred_default, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    # 绘制理想预测线
    all_y_default = np.concatenate([y_train_orig, y_test_orig])
    all_y_pred_default = np.concatenate([y_train_pred_default, y_test_pred_default])
    min_val_default = min(all_y_default.min(), all_y_pred_default.min())
    max_val_default = max(all_y_default.max(), all_y_pred_default.max())
    ax1.plot([min_val_default, max_val_default], [min_val_default, max_val_default], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax1.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax1.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax1.set_title('Default Parameters - ElasticNet', fontsize=8, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 加粗坐标轴数字
    ax1.tick_params(axis='both', which='major', labelsize=6, width=1.0)
    
    # 图例放在左上角
    ax1.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 性能标签放在右下角
    train_r2_default = r2_score(y_train_orig, y_train_pred_default)
    test_r2_default = r2_score(y_test_orig, y_test_pred_default)
    ax1.text(0.95, 0.05, f'Training R² = {train_r2_default:.3f}\nTest R² = {test_r2_default:.3f}', 
             transform=ax1.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    # 绘制调优后模型的散点图
    # 训练集点
    ax2.scatter(y_train_orig, y_train_pred, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    # 测试集点
    ax2.scatter(y_test_orig, y_test_pred, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    # 绘制理想预测线
    all_y_best = np.concatenate([y_train_orig, y_test_orig])
    all_y_pred_best = np.concatenate([y_train_pred, y_test_pred])
    min_val_best = min(all_y_best.min(), all_y_pred_best.min())
    max_val_best = max(all_y_best.max(), all_y_pred_best.max())
    ax2.plot([min_val_best, max_val_best], [min_val_best, max_val_best], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax2.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax2.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax2.set_title('Enet', fontsize=8, fontweight='bold')  # 修改标题为Enet
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 加粗坐标轴数字
    ax2.tick_params(axis='both', which='major', labelsize=6, width=1.0)
    
    # 图例放在左上角
    ax2.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 性能标签放在右下角
    train_r2_best = r2_score(y_train_orig, y_train_pred)
    test_r2_best = r2_score(y_test_orig, y_test_pred)
    ax2.text(0.95, 0.05, f'Training R² = {train_r2_best:.3f}\nTest R² = {test_r2_best:.3f}', 
             transform=ax2.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存对比图片
    plt.savefig(f'results_enet_only/{model_name}_comparison_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results_enet_only/{model_name}_comparison_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # 保存默认模型的单独图
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_train_orig, y_train_pred_default, alpha=0.8, s=25, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test_orig, y_test_pred_default, alpha=0.8, s=25, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    ax.plot([min_val_default, max_val_default], [min_val_default, max_val_default], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    ax.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax.set_title('Default ElasticNet - Observed vs Predicted', fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 加粗坐标轴数字
    ax.tick_params(axis='both', which='major', labelsize=6, width=1.0)
    
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    ax.text(0.95, 0.05, f'Training R² = {train_r2_default:.3f}\nTest R² = {test_r2_default:.3f}', 
             transform=ax.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    plt.savefig(f'results_enet_only/{model_name}_default_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results_enet_only/{model_name}_default_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # 保存调优后模型的单独图到enet_results文件夹 - 这是您要求的主要修改
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(y_train_orig, y_train_pred, alpha=0.8, s=25, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test_orig, y_test_pred, alpha=0.8, s=25, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    ax.plot([min_val_best, max_val_best], [min_val_best, max_val_best], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    ax.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax.set_title('Enet', fontsize=8, fontweight='bold')  # 修改标题为Enet
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 加粗坐标轴数字
    ax.tick_params(axis='both', which='major', labelsize=6, width=1.0)
    
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    ax.text(0.95, 0.05, f'Training R² = {train_r2_best:.3f}\nTest R² = {test_r2_best:.3f}', 
             transform=ax.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    # 保存到enet_results文件夹
    plt.savefig(f'enet_results/{model_name}_tuned_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'enet_results/{model_name}_tuned_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"对比散点图已保存至 results_enet_only/{model_name}_comparison_scatter_plot.png")
    logger.info(f"默认模型散点图已保存至 results_enet_only/{model_name}_default_scatter_plot.png")
    logger.info(f"调优模型散点图已保存至 enet_results/{model_name}_tuned_scatter_plot.png")
    
    # 返回默认模型的预测结果，用于绘制残差图
    return y_train_pred_default, y_test_pred_default

def save_scatter_data(y_train, y_train_pred, y_test, y_test_pred, X_train, X_test, y_train_orig, y_test_orig, feature_cols, model_name):
    """保存散点图数据到Excel文件"""
    os.makedirs('enet_results', exist_ok=True)
    
    # 重新训练默认参数ElasticNet模型以获取预测值
    default_enet_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(random_state=42))  # 使用默认参数
    ])
    default_enet_pipe.fit(X_train, y_train_orig)
    
    # 获取默认参数的预测
    y_train_pred_default = default_enet_pipe.predict(X_train)
    y_test_pred_default = default_enet_pipe.predict(X_test)
    
    # 创建DataFrame保存数据
    scatter_data = pd.DataFrame({
        'Experimental_Train': y_train_orig.values,
        'Predicted_Default_Train': y_train_pred_default,
        'Predicted_Best_Train': y_train_pred,
        'Set_Type': 'Training',
        'Index': y_train_orig.index
    })
    
    test_data = pd.DataFrame({
        'Experimental_Test': y_test_orig.values,
        'Predicted_Default_Test': y_test_pred_default,
        'Predicted_Best_Test': y_test_pred,
        'Set_Type': 'Test',
        'Index': y_test_orig.index
    })
    
    # 合并训练和测试数据
    full_data = pd.concat([
        scatter_data.rename(columns={
            'Experimental_Train': 'Experimental',
            'Predicted_Default_Train': 'Predicted_Default',
            'Predicted_Best_Train': 'Predicted_Best'
        }),
        test_data.rename(columns={
            'Experimental_Test': 'Experimental',
            'Predicted_Default_Test': 'Predicted_Default',
            'Predicted_Best_Test': 'Predicted_Best'
        })
    ], ignore_index=True)
    
    # 保存到Excel
    with pd.ExcelWriter(f'enet_results/{model_name}_scatter_plot_data.xlsx') as writer:
        full_data.to_excel(writer, sheet_name='All_Data', index=False)
        
        # 分别保存训练集和测试集
        scatter_data.to_excel(writer, sheet_name='Training_Set', index=False)
        test_data.to_excel(writer, sheet_name='Test_Set', index=False)
        
        # 保存性能指标
        train_r2_default = r2_score(y_train_orig, y_train_pred_default)
        test_r2_default = r2_score(y_test_orig, y_test_pred_default)
        train_r2_best = r2_score(y_train_orig, y_train_pred)
        test_r2_best = r2_score(y_test_orig, y_test_pred)
        
        metrics_data = {
            'Metric': ['R2', 'MSE', 'RMSE', 'MAE'],
            'Default_Train': [
                train_r2_default,
                mean_squared_error(y_train_orig, y_train_pred_default),
                np.sqrt(mean_squared_error(y_train_orig, y_train_pred_default)),
                mean_absolute_error(y_train_orig, y_train_pred_default)
            ],
            'Default_Test': [
                test_r2_default,
                mean_squared_error(y_test_orig, y_test_pred_default),
                np.sqrt(mean_squared_error(y_test_orig, y_test_pred_default)),
                mean_absolute_error(y_test_orig, y_test_pred_default)
            ],
            'Best_Train': [
                train_r2_best,
                mean_squared_error(y_train_orig, y_train_pred),
                np.sqrt(mean_squared_error(y_train_orig, y_train_pred)),
                mean_absolute_error(y_train_orig, y_train_pred)
            ],
            'Best_Test': [
                test_r2_best,
                mean_squared_error(y_test_orig, y_test_pred),
                np.sqrt(mean_squared_error(y_test_orig, y_test_pred)),
                mean_absolute_error(y_test_orig, y_test_pred)
            ]
        }
        pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Performance_Metrics', index=False)
    
    logger.info(f"散点图数据已保存至 enet_results/{model_name}_scatter_plot_data.xlsx")

def _calculate_average_feature_distance(train_features, other_features=None):
    """计算平均特征距离"""
    if other_features is None:
        distance_matrix = pairwise_distances(train_features, metric='euclidean')
        np.fill_diagonal(distance_matrix, np.nan)
        return np.nanmean(distance_matrix, axis=1)
    distance_matrix = pairwise_distances(other_features, train_features, metric='euclidean')
    return distance_matrix.mean(axis=1)

def plot_applicability_domain(model, X_train, X_test, y_train, y_test):
    """绘制调优模型适用域分析图并保存数据 - 已调整为与参考代码一致的风格"""
    os.makedirs('enet_results', exist_ok=True)

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

    # 调整为参考代码中的图形大小
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
    plt.savefig('enet_results/elasticnet_applicability_domain.png', dpi=300, bbox_inches='tight')
    plt.savefig('enet_results/elasticnet_applicability_domain.tif', dpi=300, bbox_inches='tight', format='tiff')
    plt.close()

    testing_outliers = ((ad_df['Set'] == 'Testing') & ad_df['OutOfDomain']).sum()
    summary_df = pd.DataFrame({
        'Metric': ['Testing Samples Outside Domain'],
        'Count': [testing_outliers]
    })

    with pd.ExcelWriter('enet_results/elasticnet_applicability_domain_data.xlsx') as writer:
        ad_df.to_excel(writer, sheet_name='All_Samples', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    logger.info("弹性网络适用域分析已保存至 enet_results/ 目录")

def plot_residuals(y_train, y_train_pred, y_test, y_test_pred, y_train_pred_default, y_test_pred_default, model_name):
    """绘制残差图 - SCI风格，包含默认参数和调优后模型的对比"""
    os.makedirs('results_enet_only', exist_ok=True)
    
    # 创建包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # 定义清晰的颜色
    train_color = '#FF0000'  # 红色
    test_color = '#0000FF'   # 蓝色
    line_color = '#45B7D1'   # 青色
    
    # 计算默认模型的残差
    train_residuals_default = y_train - y_train_pred_default
    test_residuals_default = y_test - y_test_pred_default
    
    # 绘制默认模型的残差图
    ax1.scatter(y_train_pred_default, train_residuals_default, alpha=0.8, s=20, color=train_color, 
                edgecolors='white', linewidth=0.5, label='Training set')
    ax1.scatter(y_test_pred_default, test_residuals_default, alpha=0.8, s=20, color=test_color, 
                edgecolors='white', linewidth=0.5, label='Test set')
    
    # 零线
    ax1.axhline(y=0, color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax1.set_xlabel('Predicted log EC50', fontsize=7)
    ax1.set_ylabel('Residuals', fontsize=7)
    ax1.set_title('Default Parameters - Residual Plot', fontsize=8, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 计算调优后模型的残差
    train_residuals_best = y_train - y_train_pred
    test_residuals_best = y_test - y_test_pred
    
    # 绘制调优后模型的残差图
    ax2.scatter(y_train_pred, train_residuals_best, alpha=0.8, s=20, color=train_color, 
                edgecolors='white', linewidth=0.5, label='Training set')
    ax2.scatter(y_test_pred, test_residuals_best, alpha=0.8, s=20, color=test_color, 
                edgecolors='white', linewidth=0.5, label='Test set')
    
    # 零线
    ax2.axhline(y=0, color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax2.set_xlabel('Predicted log EC50', fontsize=7)
    ax2.set_ylabel('Residuals', fontsize=7)
    ax2.set_title('Tuned Parameters - Residual Plot', fontsize=8, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存对比残差图
    plt.savefig(f'results_enet_only/{model_name}_comparison_residual_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results_enet_only/{model_name}_comparison_residual_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # 保存默认模型的单独残差图
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(y_train_pred_default, train_residuals_default, alpha=0.8, s=25, color=train_color, 
                edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test_pred_default, test_residuals_default, alpha=0.8, s=25, color=test_color, 
                edgecolors='white', linewidth=0.5, label='Test set')
    ax.axhline(y=0, color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    ax.set_xlabel('Predicted log EC50', fontsize=7)
    ax.set_ylabel('Residuals', fontsize=7)
    ax.set_title('Default ElasticNet - Residual Plot', fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    plt.savefig(f'results_enet_only/{model_name}_default_residual_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results_enet_only/{model_name}_default_residual_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # 保存调优后模型的单独残差图
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(y_train_pred, train_residuals_best, alpha=0.8, s=25, color=train_color, 
                edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test_pred, test_residuals_best, alpha=0.8, s=25, color=test_color, 
                edgecolors='white', linewidth=0.5, label='Test set')
    ax.axhline(y=0, color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    ax.set_xlabel('Predicted log EC50', fontsize=7)
    ax.set_ylabel('Residuals', fontsize=7)
    ax.set_title('Tuned ElasticNet - Residual Plot', fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    plt.savefig(f'results_enet_only/{model_name}_tuned_residual_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results_enet_only/{model_name}_tuned_residual_plot.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"对比残差图已保存至 results_enet_only/{model_name}_comparison_residual_plot.png")
    logger.info(f"默认模型残差图已保存至 results_enet_only/{model_name}_default_residual_plot.png")
    logger.info(f"调优模型残差图已保存至 results_enet_only/{model_name}_tuned_residual_plot.png")

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

    legend_font = FontProperties(size=9, weight='bold')
    plt.legend(
        loc='upper right',
        prop=legend_font,
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

def plot_model_performance(results, best_params):
    """绘制模型性能比较图并保存结果"""
    os.makedirs('results_enet_only', exist_ok=True)
    
    # 保存结果到CSV
    train_metrics = {k: v['train'] for k, v in results.items()}
    test_metrics = {k: v['test'] for k, v in results.items()}
    
    metrics_df = pd.DataFrame(train_metrics).T
    metrics_df.to_csv('results_enet_only/train_performance.csv')
    
    test_metrics_df = pd.DataFrame(test_metrics).T
    test_metrics_df.to_csv('results_enet_only/test_performance.csv')
    
    # 保存最佳参数
    params_df = pd.DataFrame(best_params).T
    params_df.to_csv('results_enet_only/best_params.csv')
    
    # 绘制性能比较图
    plt.figure(figsize=(12, 8))
    metrics_df[['RMSE', 'MAE']].plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('ElasticNet Model Performance (RMSE and MAE) - Training Set')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results_enet_only/model_performance_comparison.png', dpi=300)
    plt.close()
    
    # 绘制R²比较图
    plt.figure(figsize=(10, 6))
    metrics_df['R2'].plot(kind='bar', color='lightgreen')
    plt.title('ElasticNet Model R² Score - Training Set')
    plt.ylabel('R² Value')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results_enet_only/model_r2_comparison.png', dpi=300)
    plt.close()

def plot_feature_importance(models, feature_cols, X_train):
    """绘制特征重要性图"""
    os.makedirs('feature_importance_enet_only', exist_ok=True)
    
    for model_name, model in models.items():
        try:
            logger.info(f"\n使用SHAP分析 {model_name} 的特征重要性...")
            
            # 获取预处理后的训练数据
            if hasattr(model, 'named_steps'):
                # 对于Pipeline，获取预处理后的数据
                X_processed = model.named_steps['scaler'].transform(
                    model.named_steps['feature_selector'].transform(X_train)
                )
                # 获取最终估计器
                final_estimator = model.named_steps[model.steps[-1][0]]
            else:
                X_processed = X_train
                final_estimator = model
            
            # 创建解释器
            if hasattr(final_estimator, 'coef_'):
                # 线性模型
                explainer = shap.LinearExplainer(final_estimator, X_processed)
                
                # 计算SHAP值 (使用小样本加快计算)
                sample_idx = np.random.choice(range(len(X_processed)), min(100, len(X_processed)), replace=False)
                X_sample = X_processed[sample_idx]
                shap_values = explainer.shap_values(X_sample)
                
                # 绘制摘要图
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, 
                                 plot_type="bar", show=False)
                plt.title(f'{model_name} - SHAP Feature Importance')
                plt.tight_layout()
                
                img_file = f'feature_importance_enet_only/{model_name}_shap_importance.png'
                plt.savefig(img_file, dpi=300)
                plt.close()
                logger.info(f"SHAP特征重要性图已保存至: {img_file}")
                
        except Exception as e:
            logger.error(f"绘制 {model_name} 特征重要性时出错: {str(e)}")

def save_predictions_to_excel(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, 
                             feature_cols, original_data, best_params, results):
    """保存最优参数模型的预测结果到Excel表格"""
    os.makedirs('enet_results', exist_ok=True)
    
    # 创建训练集预测结果DataFrame
    train_results = pd.DataFrame({
        'Experimental_values': y_train.values,
        'Predicted_values': y_train_pred,
        'Residuals': y_train.values - y_train_pred
    }, index=y_train.index)
    
    # 创建测试集预测结果DataFrame
    test_results = pd.DataFrame({
        'Experimental_values': y_test.values,
        'Predicted_values': y_test_pred,
        'Residuals': y_test.values - y_test_pred
    }, index=y_test.index)
    
    # 合并原始数据中的相关信息
    if 'Name' in original_data.columns:
        train_results = train_results.join(original_data.loc[train_results.index, ['Name']])
        test_results = test_results.join(original_data.loc[test_results.index, ['Name']])
    
    if 'Canonical SMILES' in original_data.columns:
        train_results = train_results.join(original_data.loc[train_results.index, ['Canonical SMILES']])
        test_results = test_results.join(original_data.loc[test_results.index, ['Canonical SMILES']])
    
    # 创建性能指标DataFrame
    metrics_df = pd.DataFrame({
        'Training': results['ElasticNet']['train'],
        'Test': results['ElasticNet']['test']
    })
    
    # 创建参数DataFrame
    params_df = pd.DataFrame(list(best_params['ElasticNet'].items()), columns=['Parameter', 'Value'])
    
    # 保存到Excel文件
    with pd.ExcelWriter('enet_results/best_model_predictions.xlsx') as writer:
        train_results.to_excel(writer, sheet_name='Training Set Predictions', index=True)
        test_results.to_excel(writer, sheet_name='Test Set Predictions', index=True)
        metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=True)
        params_df.to_excel(writer, sheet_name='Model Parameters', index=False)
    
    logger.info("最优参数模型的预测结果已保存至 enet_results/best_model_predictions.xlsx")

def main():
    try:
        input_file = "IPC-81_molecular_descriptors_reduced.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info(f"开始处理文件: {input_file}")
        start_time = time.time()
        
        # 加载和准备数据
        X_train, X_test, y_train, y_test, feature_cols, original_data = load_and_prepare_data(input_file)
        
        # 训练和调优模型
        results, models, best_params, y_train_pred, y_test_pred = train_and_tune_models(
            X_train, X_test, y_train, y_test, feature_cols
        )
        
        # 打印模型性能
        logger.info("\n=== 模型性能 ===")
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name}性能:")
            logger.info(f"  训练集 - MSE: {metrics['train']['MSE']:.4f}, RMSE: {metrics['train']['RMSE']:.4f}, MAE: {metrics['train']['MAE']:.4f}, R²: {metrics['train']['R2']:.4f}, MRE (%): {metrics['train']['MRE (%)']:.4f}")
            logger.info(f"  测试集 - MSE: {metrics['test']['MSE']:.4f}, RMSE: {metrics['test']['RMSE']:.4f}, MAE: {metrics['test']['MAE']:.4f}, R²: {metrics['test']['R2']:.4f}, MRE (%): {metrics['test']['MRE (%)']:.4f}")
        
        # 选择最佳模型
        best_model_name = min(results, key=lambda k: results[k]['test']['RMSE'])
        best_model = models[best_model_name]
        logger.info(f"\n最佳模型: {best_model_name} (测试集RMSE: {results[best_model_name]['test']['RMSE']:.4f})")
        
        # 绘制散点图 - 获取默认模型的预测结果
        y_train_pred_default, y_test_pred_default = plot_scatter(y_train, y_train_pred, y_test, y_test_pred, "ElasticNet", 
                    X_train, X_test, y_train, y_test, feature_cols)
        
        # 绘制残差图 - 传递默认模型的预测结果
        plot_residuals(y_train, y_train_pred, y_test, y_test_pred, y_train_pred_default, y_test_pred_default, "ElasticNet")

        # 绘制残差直方图（仅调优模型测试集）
        tuned_test_residuals = (y_test.values - y_test_pred).astype(float)
        plot_residual_histogram(tuned_test_residuals, "ElasticNet", "results_enet_only")
        
        # 绘制特征重要性
        plot_feature_importance(models, feature_cols, X_train)
        
        # 绘制模型性能比较
        plot_model_performance(results, best_params)
        
        # 保存最优参数模型的预测结果到Excel
        save_predictions_to_excel(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, 
                                feature_cols, original_data, best_params, results)
        # 保存散点图数据
        save_scatter_data(y_train, y_train_pred, y_test, y_test_pred, X_train, X_test, 
                          y_train, y_test, feature_cols, "ElasticNet")
        # 绘制适用域分析（使用调优后的最佳模型）
        plot_applicability_domain(best_model, X_train, X_test, y_train, y_test)
        # 保存最佳模型
        joblib.dump(best_model, f'best_model_{best_model_name}.pkl')
        logger.info(f"最佳模型已保存为: best_model_{best_model_name}.pkl")
        
        # 执行时间
        elapsed = time.time() - start_time
        logger.info(f"\n总执行时间: {elapsed:.2f}秒")
        logger.info("处理完成!")
        
        return results, models, best_params
    
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise

if __name__ == "__main__":
    import shap
    results, models, best_params = main()