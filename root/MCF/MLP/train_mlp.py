import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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

def train_and_tune_mlp(X_train, X_test, y_train, y_test, feature_cols):
    """训练和调优MLP模型"""
    # 使用预设的默认参数
    default_params = {
        'hidden_layer_sizes': (100,100),
        'activation': 'relu',
        'alpha': 0.001,
        'learning_rate_init': 0.001,
        'batch_size': 64,
        'max_iter': 1000
    }
    
    # 创建默认模型的Pipeline
    default_pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=42, **default_params))
    ])
    
    # 训练默认模型
    default_pipeline.fit(X_train, y_train)
    
    # 训练集和测试集预测
    y_train_pred_default = default_pipeline.predict(X_train)
    y_test_pred_default = default_pipeline.predict(X_test)
    
    # 评估默认模型
    default_train_metrics = evaluate_model(y_train, y_train_pred_default)
    default_test_metrics = evaluate_model(y_test, y_test_pred_default)
    
    # 记录默认参数
    default_full_params = default_pipeline.named_steps['mlp'].get_params()
    
    # 创建调优模型的Pipeline
    mlp_pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=42, early_stopping=True, n_iter_no_change=10))
    ])
    
    # 参数空间
    param_grid = {
        'mlp__hidden_layer_sizes': [(100,), (50,50), (100,50)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01],
        'mlp__batch_size': [16, 32],
        'mlp__max_iter': [1000]
    }
    
    logger.info("开始网格搜索...")
    start_time = time.time()
    
    # 网格搜索
    grid_search = GridSearchCV(
        mlp_pipeline,
        param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"网格搜索完成，耗时: {time.time() - start_time:.2f}秒")
    logger.info(f"最佳参数: {grid_search.best_params_}")
    logger.info(f"最佳交叉验证负MSE: {grid_search.best_score_:.4f}")
    
    # 最佳模型
    best_mlp = grid_search.best_estimator_
    
    # 训练集和测试集预测
    y_train_pred_best = best_mlp.predict(X_train)
    y_test_pred_best = best_mlp.predict(X_test)
    
    # 评估最佳模型
    best_train_metrics = evaluate_model(y_train, y_train_pred_best)
    best_test_metrics = evaluate_model(y_test, y_test_pred_best)
    
    # 记录最佳参数
    best_params = best_mlp.named_steps['mlp'].get_params()
    
    # 保存结果
    results = {
        'default_train': default_train_metrics,
        'default_test': default_test_metrics,
        'best_train': best_train_metrics,
        'best_test': best_test_metrics,
        'best_params': grid_search.best_params_,
        'param_grid': param_grid,
        'cv_results': grid_search.cv_results_,
        'default_params': default_full_params,
        'best_full_params': best_params
    }
    
    return results, default_pipeline, best_mlp

def plot_scatter_comparison(X_train, X_test, y_train, y_test, feature_cols, results):
    """绘制默认参数和调优后模型的散点图对比"""
    os.makedirs('mlp_results', exist_ok=True)
    
    # 重新创建默认模型Pipeline
    default_params_copy = results['default_params'].copy()
    if 'random_state' in default_params_copy:
        del default_params_copy['random_state']
    
    default_pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=42, **default_params_copy))
    ])
    default_pipeline.fit(X_train, y_train)
    
    y_train_pred_default = default_pipeline.predict(X_train)
    y_test_pred_default = default_pipeline.predict(X_test)
    
    # 创建最佳模型Pipeline
    best_pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(** results['best_full_params']))
    ])
    best_pipeline.fit(X_train, y_train)
    y_train_pred_best = best_pipeline.predict(X_train)
    y_test_pred_best = best_pipeline.predict(X_test)
    
    # 定义颜色
    train_color = '#FF0000'  # 红色
    test_color = '#0000FF'   # 蓝色
    line_color = '#45B7D1'   # 青色
    
    # 1. 绘制对比图（两个子图）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # 默认模型散点图
    ax1.scatter(y_train, y_train_pred_default, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax1.scatter(y_test, y_test_pred_default, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    # 绘制理想预测线
    all_y_default = np.concatenate([y_train, y_test])
    all_y_pred_default = np.concatenate([y_train_pred_default, y_test_pred_default])
    min_val_default = min(all_y_default.min(), all_y_pred_default.min())
    max_val_default = max(all_y_default.max(), all_y_pred_default.max())
    ax1.plot([min_val_default, max_val_default], [min_val_default, max_val_default], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    # 修改坐标轴标签 - 加粗字体
    ax1.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax1.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax1.set_title('Default Parameters - MLP', fontsize=8, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴刻度标签加粗
    for tick in ax1.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 性能标签
    train_r2_default = r2_score(y_train, y_train_pred_default)
    test_r2_default = r2_score(y_test, y_test_pred_default)
    ax1.text(0.95, 0.05, f'Training R² = {train_r2_default:.3f}\nTest R² = {test_r2_default:.3f}', 
             transform=ax1.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    # 调优模型散点图
    ax2.scatter(y_train, y_train_pred_best, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax2.scatter(y_test, y_test_pred_best, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    # 绘制理想预测线
    all_y_best = np.concatenate([y_train, y_test])
    all_y_pred_best = np.concatenate([y_train_pred_best, y_test_pred_best])
    min_val_best = min(all_y_best.min(), all_y_pred_best.min())
    max_val_best = max(all_y_best.max(), all_y_pred_best.max())
    ax2.plot([min_val_best, max_val_best], [min_val_best, max_val_best], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    # 修改坐标轴标签 - 加粗字体
    ax2.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax2.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax2.set_title('Tuned Parameters - MLP', fontsize=8, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴刻度标签加粗
    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 性能标签
    train_r2_best = r2_score(y_train, y_train_pred_best)
    test_r2_best = r2_score(y_test, y_test_pred_best)
    ax2.text(0.95, 0.05, f'Training R² = {train_r2_best:.3f}\nTest R² = {test_r2_best:.3f}', 
             transform=ax2.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    # 添加图例
    ax1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8, frameon=True, borderpad=0.15, labelspacing=0.1, handlelength=0.9, handletextpad=0.25)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8, frameon=True, borderpad=0.15, labelspacing=0.1, handlelength=0.9, handletextpad=0.25)
    
    plt.tight_layout()
    plt.savefig('mlp_results/mlp_comparison_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_results/mlp_comparison_scatter.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. 单独保存默认参数模型图
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(y_train, y_train_pred_default, alpha=0.8, s=25, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test, y_test_pred_default, alpha=0.8, s=25, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    ax.plot([min_val_default, max_val_default], [min_val_default, max_val_default], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    ax.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax.set_title('Default MLP', fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴刻度标签加粗
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8, frameon=True, borderpad=0.15, labelspacing=0.1, handlelength=0.9, handletextpad=0.25)
    ax.text(0.95, 0.05, f'Training R² = {train_r2_default:.3f}\nTest R² = {test_r2_default:.3f}', 
             transform=ax.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    plt.savefig('mlp_results/mlp_default_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_results/mlp_default_scatter.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. 单独保存调优模型图
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(y_train, y_train_pred_best, alpha=0.8, s=25, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test, y_test_pred_best, alpha=0.8, s=25, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    ax.plot([min_val_best, max_val_best], [min_val_best, max_val_best], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    ax.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax.set_title('MLP', fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴刻度标签加粗
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8, frameon=True, borderpad=0.15, labelspacing=0.1, handlelength=0.9, handletextpad=0.25)
    ax.text(0.95, 0.05, f'Training R² = {train_r2_best:.3f}\nTest R² = {test_r2_best:.3f}', 
             transform=ax.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    plt.savefig('mlp_results/mlp_tuned_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_results/mlp_tuned_scatter.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("散点图已保存至 mlp_results/ 目录")

def save_scatter_data(X_train, X_test, y_train, y_test, feature_cols, results):
    """保存散点图数据到Excel文件"""
    os.makedirs('mlp_results', exist_ok=True)
    
    # 重新创建默认模型Pipeline
    default_params_copy = results['default_params'].copy()
    if 'random_state' in default_params_copy:
        del default_params_copy['random_state']
    
    default_pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=42, **default_params_copy))
    ])
    default_pipeline.fit(X_train, y_train)
    
    # 默认模型预测
    y_train_pred_default = default_pipeline.predict(X_train)
    y_test_pred_default = default_pipeline.predict(X_test)
    
    # 创建最佳模型Pipeline
    best_pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(** results['best_full_params']))
    ])
    best_pipeline.fit(X_train, y_train)
    
    # 最佳模型预测
    y_train_pred_best = best_pipeline.predict(X_train)
    y_test_pred_best = best_pipeline.predict(X_test)
    
    # 创建DataFrame保存数据
    scatter_data = pd.DataFrame({
        'Experimental_Train': y_train.values,
        'Predicted_Default_Train': y_train_pred_default,
        'Predicted_Best_Train': y_train_pred_best,
        'Set_Type': 'Training',
        'Index': y_train.index
    })
    
    test_data = pd.DataFrame({
        'Experimental_Test': y_test.values,
        'Predicted_Default_Test': y_test_pred_default,
        'Predicted_Best_Test': y_test_pred_best,
        'Set_Type': 'Test',
        'Index': y_test.index
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
    with pd.ExcelWriter('mlp_results/scatter_plot_data.xlsx') as writer:
        full_data.to_excel(writer, sheet_name='All_Data', index=False)
        scatter_data.to_excel(writer, sheet_name='Training_Set', index=False)
        test_data.to_excel(writer, sheet_name='Test_Set', index=False)
        
        # 保存性能指标
        metrics_data = {
            'Metric': ['MSE', 'RMSE', 'MAE', 'R2', 'MRE'],
            'Default_Train': [results['default_train'][m] for m in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']],
            'Default_Test': [results['default_test'][m] for m in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']],
            'Best_Train': [results['best_train'][m] for m in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']],
            'Best_Test': [results['best_test'][m] for m in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']]
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
    """绘制调优模型适用域分析图并保存数据（与参考代码样式一致）"""
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

    # 调整图片尺寸为参考代码样式
    fig, ax = plt.subplots(figsize=(6, 5))

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

    ax.axhline(residual_threshold, color='#d62728', linestyle='--', linewidth=0.9)
    ax.axhline(-residual_threshold, color='#d62728', linestyle='--', linewidth=0.9)
    ax.axvline(ad_threshold, color='#d62728', linestyle='--', linewidth=0.9)

    # 轴标签、标题、刻度文字设置（与参考代码一致的字体大小和加粗）
    ax.set_xlabel('Average Feature Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standardized Residuals', fontsize=12, fontweight='bold')
    ax.set_title('Applicability Domain Analysis (Training & Testing)', fontsize=12, fontweight='bold')

    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')

    # 图例样式（完全参考代码设置）
    ax.legend(
        fontsize=12,          # 图例字体大小
        loc='upper right',    # 图例位置
        framealpha=0.9,       # 图例背景透明度
        fancybox=True,        # 圆角边框
        shadow=False,         # 去掉阴影
        borderpad=0.5,        # 图例内边距
        labelspacing=0.4,     # 图例项间距
        handlelength=0.1,     # 图例标记长度
        handletextpad=0.6,    # 标记与文字间距
        columnspacing=0.5,    # 多列间距
        prop={'weight': 'bold', 'size': 11}  # 图例文字加粗
    )
    ax.grid(True, linestyle='--', alpha=0.3)

    # 统计文本样式（完全参考代码设置）
    exceed_text = (
        f"Training outside: {((ad_df['Set'] == 'Training') & ad_df['OutOfDomain']).sum()}\n"
        f"Testing outside: {((ad_df['Set'] == 'Testing') & ad_df['OutOfDomain']).sum()}"
    )
    ax.text(
        0.013,                # x位置（相对坐标）
        0.983,                # y位置（相对坐标）
        exceed_text,          # 文本内容
        transform=ax.transAxes,  # 相对坐标系
        fontsize=11,          # 文本字体大小
        fontweight='bold',    # 文本加粗
        fontfamily='Arial',   # 字体
        color='darkred',      # 文本颜色
        verticalalignment='top',  # 垂直对齐
        bbox=dict(
            boxstyle='round,pad=0.3',  # 文本框样式
            facecolor='lightyellow',   # 背景色
            alpha=0.8,                 # 透明度
            edgecolor='orange',        # 边框色
            linewidth=0.9              # 边框宽度
        )
    )

    plt.tight_layout()
    # 保存为PNG和TIFF两种格式（参考代码样式）
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
    
    # 1. 模型性能比较
    metrics_df = pd.DataFrame({
        'Default': results['default_test'],
        'Tuned': results['best_test']
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
    plt.savefig('mlp_results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_results/model_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. 训练损失曲线
    best_mlp_model = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(** results['best_full_params']))
    ])
    best_mlp_model.fit(X_train, y_train)
    
    # 获取损失曲线
    loss_curve = best_mlp_model.named_steps['mlp'].loss_curve_
    
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
                'Default_Training': results['default_train'][metric],
                'Default_Test': results['default_test'][metric],
                'Tuned_Training': results['best_train'][metric],
                'Tuned_Test': results['best_test'][metric]
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_excel(writer, sheet_name='Model Performance', index=False)
        
        # 2. 训练集详细指标
        train_metrics_data = []
        for metric in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']:
            train_metrics_data.append({
                'Metric': metric,
                'Default Model': results['default_train'][metric],
                'Tuned Model': results['best_train'][metric],
                'Improvement': results['default_train'][metric] - results['best_train'][metric] if metric != 'R2' else results['best_train'][metric] - results['default_train'][metric]
            })
        
        train_metrics_df = pd.DataFrame(train_metrics_data)
        train_metrics_df.to_excel(writer, sheet_name='Training Set Metrics', index=False)
        
        # 3. 测试集详细指标
        test_metrics_data = []
        for metric in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']:
            test_metrics_data.append({
                'Metric': metric,
                'Default Model': results['default_test'][metric],
                'Tuned Model': results['best_test'][metric],
                'Improvement': results['default_test'][metric] - results['best_test'][metric] if metric != 'R2' else results['best_test'][metric] - results['default_test'][metric]
            })
        
        test_metrics_df = pd.DataFrame(test_metrics_data)
        test_metrics_df.to_excel(writer, sheet_name='Test Set Metrics', index=False)
        
        # 4. 其他结果表...
        best_params_df = pd.DataFrame([results['best_params']])
        best_params_df.to_excel(writer, sheet_name='Best Parameters', index=False)
        
        param_grid_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results['param_grid'].items()]))
        param_grid_df.to_excel(writer, sheet_name='Parameter Grid', index=False)
        
        default_arch_df = pd.DataFrame([results['default_params']])
        default_arch_df.to_excel(writer, sheet_name='Default Architecture', index=False)
        
        best_full_params_df = pd.DataFrame([results['best_full_params']])
        best_full_params_df.to_excel(writer, sheet_name='Best Full Parameters', index=False)
        
        param_comparison_data = []
        for param, values in results['param_grid'].items():
            clean_param = param.replace('mlp__', '')
            param_comparison_data.append({
                'Parameter': clean_param,
                'Tuning Range': str(values),
                'Default Value': results['default_params'].get(clean_param, 'N/A'),
                'Best Value': results['best_full_params'].get(clean_param, 'N/A')
            })
        
        param_comparison_df = pd.DataFrame(param_comparison_data)
        param_comparison_df.to_excel(writer, sheet_name='Parameter Comparison', index=False)
        
        cv_results = results['cv_results']
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_excel(writer, sheet_name='CV Results', index=False)
        
        comparison_data = []
        for metric in ['MSE', 'RMSE', 'MAE', 'R2', 'MRE']:
            comparison_data.append({
                'Metric': metric,
                'Default_Training': results['default_train'][metric],
                'Default_Test': results['default_test'][metric],
                'Default_Overfitting': results['default_train'][metric] - results['default_test'][metric] if metric != 'R2' else results['default_test'][metric] - results['default_train'][metric],
                'Tuned_Training': results['best_train'][metric],
                'Tuned_Test': results['best_test'][metric],
                'Tuned_Overfitting': results['best_train'][metric] - results['best_test'][metric] if metric != 'R2' else results['best_test'][metric] - results['best_train'][metric]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Training vs Test Comparison', index=False)

if __name__ == "__main__":
    try:
        input_file = "MCF-7_molecular_descriptors_reduced1.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info(f"开始处理文件: {input_file}")
        start_time = time.time()
        
        # 加载和准备数据
        X_train, X_test, y_train, y_test, feature_cols, df = load_and_prepare_data(input_file)
        
        # 训练和调优模型
        results, default_mlp, best_mlp = train_and_tune_mlp(X_train, X_test, y_train, y_test, feature_cols)

        # 绘制残差直方图（仅调优模型测试集）
        tuned_test_residuals = (y_test - best_mlp.predict(X_test)).to_numpy()
        plot_residual_histogram(tuned_test_residuals, 'mlp_tuned_test', 'mlp_results')
        
        # 绘制散点图对比
        plot_scatter_comparison(X_train, X_test, y_train, y_test, feature_cols, results)
        
        # 保存散点图数据到Excel 
        save_scatter_data(X_train, X_test, y_train, y_test, feature_cols, results)

        # 绘制适用域分析（使用调优最佳模型）
        plot_applicability_domain(best_mlp, X_train, X_test, y_train, y_test)
        
        # 绘制其他结果
        plot_results(results, X_train, X_test, y_train, y_test, feature_cols)
        
        # 保存结果到Excel
        save_results_to_excel(results, feature_cols)
        
        # 保存模型
        joblib.dump(default_mlp, 'mlp_results/default_mlp_model.pkl')
        joblib.dump(best_mlp, 'mlp_results/tuned_mlp_model.pkl')
        
        # 记录执行时间
        elapsed = time.time() - start_time
        logger.info(f"\n总执行时间: {elapsed:.2f}秒")
        logger.info("MLP模型调优完成!")
        
        # 打印最终结果摘要
        print("\n" + "="*80)
        print("MLP TUNING RESULTS SUMMARY")
        print("="*80)
        print("Default Model Performance:")
        print(f"  Training Set - MSE: {results['default_train']['MSE']:.4f}, RMSE: {results['default_train']['RMSE']:.4f}, MAE: {results['default_train']['MAE']:.4f}, R²: {results['default_train']['R2']:.4f}, MRE: {results['default_train']['MRE']:.4f}")
        print(f"  Test Set - MSE: {results['default_test']['MSE']:.4f}, RMSE: {results['default_test']['RMSE']:.4f}, MAE: {results['default_test']['MAE']:.4f}, R²: {results['default_test']['R2']:.4f}, MRE: {results['default_test']['MRE']:.4f}")
        print("\nTuned Model Performance:")
        print(f"  Training Set - MSE: {results['best_train']['MSE']:.4f}, RMSE: {results['best_train']['RMSE']:.4f}, MAE: {results['best_train']['MAE']:.4f}, R²: {results['best_train']['R2']:.4f}, MRE: {results['best_train']['MRE']:.4f}")
        print(f"  Test Set - MSE: {results['best_test']['MSE']:.4f}, RMSE: {results['best_test']['RMSE']:.4f}, MAE: {results['best_test']['MAE']:.4f}, R²: {results['best_test']['R2']:.4f}, MRE: {results['best_test']['MRE']:.4f}")
        print("\nImprovement:")
        print(f"  Training Set - MSE: {results['default_train']['MSE'] - results['best_train']['MSE']:.4f}, R²: {results['best_train']['R2'] - results['default_train']['R2']:.4f}")
        print(f"  Test Set - MSE: {results['default_test']['MSE'] - results['best_test']['MSE']:.4f}, R²: {results['best_test']['R2'] - results['default_test']['R2']:.4f}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise