import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
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
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

# 设置matplotlib参数，适合SCI发表
plt.rcParams.update({
    'font.size': 9,           # 基础字体大小
    'axes.titlesize': 10,      # 子图标题字体大小
    'axes.labelsize': 10,      # 轴标签字体大小
    'xtick.labelsize': 9,     # x轴刻度标签字体大小
    'ytick.labelsize': 9,     # y轴刻度标签字体大小
    'legend.fontsize': 8,     # 图例默认字体大小
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

# 设置日志 - 解决编码问题
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("svr_optimization.log", encoding='utf-8'),
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
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    except Exception as e:
        logger.error(f"数据准备错误: {str(e)}")
        raise

def optimize_svr_model(X_train, y_train, feature_cols):
    """优化SVR模型"""
    # SVR参数优化
    svr_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    
    param_grid = {
        'svr__kernel': ['rbf'],
        'svr__C': [1.2, 1.35, 1.5, 1.65, 1.8],
        'svr__gamma': ['scale', 0.03, 0.05, 0.07],
        'svr__epsilon': [0.03, 0.04, 0.05, 0.06, 0.07]
    }
    
    logger.info("开始优化SVR模型（默认参数邻域网格搜索）...")
    svr_search = GridSearchCV(
        svr_pipe,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    svr_search.fit(X_train, y_train)
    
    logger.info(f"SVR最佳参数: {svr_search.best_params_}")
    logger.info(f"SVR最佳交叉验证R2: {svr_search.best_score_:.4f}")
    
    return svr_search.best_estimator_, svr_search

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
    os.makedirs('svr_results', exist_ok=True)
    
    # 训练默认参数SVR模型
    default_svr_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('svr', SVR())  # 使用默认参数
    ])
    default_svr_pipe.fit(X_train, y_train_orig)
    
    # 获取默认参数的预测
    y_train_pred_default = default_svr_pipe.predict(X_train)
    y_test_pred_default = default_svr_pipe.predict(X_test)
    
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
    
    # 修改横纵坐标标签，加粗字体
    ax1.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax1.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax1.set_title('Default Parameters - SVR', fontsize=8, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 设置坐标轴刻度标签加粗
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(6)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(6)
    
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
    
    # 修改横纵坐标标签，加粗字体
    ax2.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax2.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax2.set_title('SVR', fontsize=8, fontweight='bold')  # 修改标题为"SVR"
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 设置坐标轴刻度标签加粗
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(6)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(6)
    
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
    plt.savefig(f'svr_results/{model_name}_comparison_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'svr_results/{model_name}_comparison_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # 另外保存单独的调优后模型图
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(y_train_orig, y_train_pred, alpha=0.8, s=25, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test_orig, y_test_pred, alpha=0.8, s=25, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    ax.plot([min_val_best, max_val_best], [min_val_best, max_val_best], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    # 修改横纵坐标标签，加粗字体
    ax.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax.set_title('SVR', fontsize=8, fontweight='bold')  # 修改标题为"SVR"
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 设置坐标轴刻度标签加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(6)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(6)
    
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    ax.text(0.95, 0.05, f'Training R² = {train_r2_best:.3f}\nTest R² = {test_r2_best:.3f}', 
             transform=ax.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    plt.savefig(f'svr_results/{model_name}_tuned_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'svr_results/{model_name}_tuned_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"对比散点图已保存至 svr_results/{model_name}_comparison_scatter_plot.png")
    logger.info(f"单独调优散点图已保存至 svr_results/{model_name}_tuned_scatter_plot.png")
    
    # 返回默认模型的性能指标用于比较
    default_metrics = {
        'train': {
            'R2': train_r2_default,
            'MSE': mean_squared_error(y_train_orig, y_train_pred_default),
            'RMSE': np.sqrt(mean_squared_error(y_train_orig, y_train_pred_default)),
            'MAE': mean_absolute_error(y_train_orig, y_train_pred_default)
        },
        'test': {
            'R2': test_r2_default,
            'MSE': mean_squared_error(y_test_orig, y_test_pred_default),
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_test_pred_default)),
            'MAE': mean_absolute_error(y_test_orig, y_test_pred_default)
        }
    }
    
    return default_metrics

def save_scatter_data(y_train, y_train_pred, y_test, y_test_pred, X_train, X_test, y_train_orig, y_test_orig, feature_cols, model_name):
    """保存散点图数据到Excel文件"""
    os.makedirs('svr_results', exist_ok=True)
    
    # 重新训练默认参数SVR模型以获取预测值
    default_svr_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('svr', SVR())  # 使用默认参数
    ])
    default_svr_pipe.fit(X_train, y_train_orig)
    
    # 获取默认参数的预测
    y_train_pred_default = default_svr_pipe.predict(X_train)
    y_test_pred_default = default_svr_pipe.predict(X_test)
    
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
    with pd.ExcelWriter(f'svr_results/{model_name}_scatter_plot_data.xlsx') as writer:
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
    
    logger.info(f"散点图数据已保存至 svr_results/{model_name}_scatter_plot_data.xlsx")

def _calculate_average_feature_distance(train_features, other_features=None):
    """计算平均特征距离"""
    if other_features is None:
        distance_matrix = pairwise_distances(train_features, metric='euclidean')
        np.fill_diagonal(distance_matrix, np.nan)
        return np.nanmean(distance_matrix, axis=1)
    distance_matrix = pairwise_distances(other_features, train_features, metric='euclidean')
    return distance_matrix.mean(axis=1)

def plot_applicability_domain(model, X_train, X_test, y_train, y_test, model_name):
    """绘制模型适用域分析图 - 调整为与参考代码一致的风格"""
    os.makedirs('svr_results', exist_ok=True)

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
    plt.savefig(f'svr_results/{model_name}_applicability_domain.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'svr_results/{model_name}_applicability_domain.tif', dpi=300, bbox_inches='tight', format='tiff')
    plt.close()

    testing_outliers = ((ad_df['Set'] == 'Testing') & ad_df['OutOfDomain']).sum()
    summary_df = pd.DataFrame({
        'Metric': ['Testing Samples Outside Domain'],
        'Count': [testing_outliers]
    })

    with pd.ExcelWriter(f'svr_results/{model_name}_applicability_domain_data.xlsx') as writer:
        ad_df.to_excel(writer, sheet_name='All_Samples', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    logger.info(f"适用域分析结果已保存至 svr_results/{model_name}_applicability_domain.*")

def plot_residuals(y_train, y_train_pred, y_test, y_test_pred, model_name):
    """绘制残差图 - SCI风格"""
    os.makedirs('svr_results', exist_ok=True)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    # 修改为红蓝对比色
    train_color = '#FF0000'  # 红色
    test_color = '#0000FF'   # 蓝色
    line_color = '#45B7D1'   # 蓝色
    
    # 计算残差
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    ax.scatter(y_train_pred, train_residuals, alpha=0.8, s=25, color=train_color, 
                edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test_pred, test_residuals, alpha=0.8, s=25, color=test_color, 
                edgecolors='white', linewidth=0.5, label='Test set')
    
    # 零线
    ax.axhline(y=0, color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax.set_xlabel('Predicted log EC50', fontsize=7)
    ax.set_ylabel('Residuals', fontsize=7)
    ax.set_title('Residual Plot', fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 图例放在左上角
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 保存图片
    plt.savefig(f'svr_results/{model_name}_residual_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'svr_results/{model_name}_residual_plot.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"残差图已保存至 svr_results/{model_name}_residual_plot.png")

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


import shap
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def create_shap_analysis(X_train, X_test, y_train, y_test, feature_cols, model_name):
    """创建SHAP分析图表 - 增强字体和横坐标轴加粗效果"""
    os.makedirs('svr_results', exist_ok=True)
    
    logger.info("开始创建SHAP分析...")
    
    # 模型训练与SHAP值计算（保持不变）
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_model.fit(X_train_scaled, y_train)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # 特征重要性计算（保持不变）
    mean_shap_abs = np.mean(np.abs(shap_values), axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_ABS_SHAP': mean_shap_abs
    }).sort_values('Mean_ABS_SHAP', ascending=True)
    top_features = feature_importance_df.tail(15)

    fig_width, fig_height = 6, 8

    # === 图(a): Global explanation - 特征重要性条形图 ===
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    bars = ax.barh(range(len(top_features)), 
                   top_features['Mean_ABS_SHAP'],
                   color=colors, alpha=0.85, edgecolor='none')
    
    # 1. Y轴特征标签：强化加粗
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(
        top_features['Feature'], 
        fontsize=12, 
        fontweight='extra bold',  # 从bold改为extra bold（更粗）
        color='#000000'
    )
    
    # 2. X轴标签：强化加粗
    ax.set_xlabel(
        'mean(|SHAP value|)', 
        fontsize=12, 
        fontweight='extra bold',  # 更粗
        color='#000000'
    )
    
    # 3. X轴刻度标签：强化加粗
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight('extra bold')  # 更粗
        label.set_color('#000000')
    
    # 4. 横坐标轴（X轴）线条加粗（从1.2→2.0）
    ax.tick_params(axis='x', width=2.0, length=5)  # 刻度线加粗
    ax.spines['bottom'].set_linewidth(2.0)  # X轴线加粗
    
    # Y轴线条和刻度保持但不弱化
    ax.tick_params(axis='y', width=1.2, length=4)
    ax.spines['left'].set_linewidth(1.2)
    
    # 隐藏顶部和右侧边框（保持）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 5. 条形图上的数值标签：强化加粗
    for i, (bar, value) in enumerate(zip(bars, top_features['Mean_ABS_SHAP'])):
        ax.text(
            value + 0.001, 
            bar.get_y() + bar.get_height()/2, 
            f'{value:.3f}', 
            ha='left', 
            va='center', 
            fontsize=12, 
            fontweight='extra bold',  # 更粗
            color='#000000'
        )
    
    # 背景和布局（保持）
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.1)
    
    # 保存图(a)
    base_path = f'svr_results/{model_name}_global_explanation'
    plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{base_path}.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(f'{base_path}.tif', dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

    # === 图(b): Local explanation - SHAP summary plot ===
    plt.figure(figsize=(fig_width, fig_height))
    shap.summary_plot(
        shap_values, 
        X_test_scaled,
        feature_names=feature_cols,
        plot_type="dot",
        show=False,
        max_display=15
    )
    
    ax = plt.gca()
    
    # 6.  summary plot X轴标签：强化加粗
    ax.set_xlabel(
        'SHAP value (impact on model output)', 
        fontsize=12, 
        fontweight='extra bold',  # 更粗
        color='#000000'
    )
    ax.set_ylabel('')  # 移除Features标签
    
    # 7. X轴刻度标签：强化加粗
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight('extra bold')  # 更粗
        label.set_color('#000000')
    
    # 8. Y轴特征标签：强化加粗
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('extra bold')  # 更粗
        label.set_color('#000000')
    
    # 9. 横坐标轴（X轴）线条和刻度加粗（从1.2→2.0）
    ax.tick_params(axis='x', width=2.0, length=5)  # 刻度线加粗
    ax.spines['bottom'].set_linewidth(2.0)  # X轴线加粗
    
    # Y轴保持
    ax.tick_params(axis='y', width=1.2, length=4)
    ax.spines['left'].set_linewidth(1.2)
    
    # 10. 颜色条标签：强化加粗
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('')
    for label in cbar.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('extra bold')  # 更粗
        label.set_color('#000000')
    
    # 背景和布局（保持）
    ax.set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    ax.grid(False)
    plt.subplots_adjust(left=0.25, right=0.85, top=0.95, bottom=0.1)
    
    # 保存图(b)
    base_path = f'svr_results/{model_name}_local_explanation'
    plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{base_path}.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(f'{base_path}.tif', dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

    # 保存Excel（保持不变）
    feature_importance_df_sorted = feature_importance_df.sort_values('Mean_ABS_SHAP', ascending=False)
    with pd.ExcelWriter('svr_results/feature_importance_analysis.xlsx') as writer:
        feature_importance_df_sorted.to_excel(writer, sheet_name='Feature Importance', index=False)
        shap_values_df = pd.DataFrame(shap_values, columns=feature_cols)
        shap_values_df.to_excel(writer, sheet_name='SHAP Values', index=False)
        shap_stats = pd.DataFrame({
            'Feature': feature_cols,
            'Mean_SHAP': np.mean(shap_values, axis=0),
            'Std_SHAP': np.std(shap_values, axis=0),
            'Min_SHAP': np.min(shap_values, axis=0),
            'Max_SHAP': np.max(shap_values, axis=0),
            'Mean_ABS_SHAP': mean_shap_abs
        }).sort_values('Mean_ABS_SHAP', ascending=False)
        shap_stats.to_excel(writer, sheet_name='SHAP Statistics', index=False)

    logger.info(f"SHAP分析图表已保存完成（含TIF格式）")
    logger.info(f"特征重要性分析已保存至 svr_results/feature_importance_analysis.xlsx")
    
    return feature_importance_df_sorted, shap_values

def save_results_to_excel(results, feature_cols, X_train, y_train, train_metrics, test_metrics):
    """保存结果到Excel文件 - 添加训练集指标"""
    os.makedirs('svr_results', exist_ok=True)
    
    # 创建Excel写入器
    with pd.ExcelWriter('svr_results/svr_results.xlsx') as writer:
        # 1. 模型性能比较
        metrics_df = pd.DataFrame({
            'Default Model': results['default'],
            'Tuned Model': results['tuned']
        }).T
        metrics_df.to_excel(writer, sheet_name='Model Performance')
        
        # 2. 训练集详细指标
        train_metrics_df = pd.DataFrame([train_metrics])
        train_metrics_df.to_excel(writer, sheet_name='Training Set Metrics', index=False)
        
        # 3. 测试集详细指标
        test_metrics_df = pd.DataFrame([test_metrics])
        test_metrics_df.to_excel(writer, sheet_name='Test Set Metrics', index=False)
        
        # 4. 最佳参数
        best_params_df = pd.DataFrame([results['best_params']])
        best_params_df.to_excel(writer, sheet_name='Best Parameters', index=False)
        
        # 5. 参数网格
        param_grid_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results['param_grid'].items()]))
        param_grid_df.to_excel(writer, sheet_name='Parameter Grid', index=False)
        
        # 6. 默认模型架构
        default_arch_df = pd.DataFrame([results['default_params']])
        default_arch_df.to_excel(writer, sheet_name='Default Architecture', index=False)
        
        # 7. 最佳模型完整参数
        best_full_params_df = pd.DataFrame([results['best_full_params']])
        best_full_params_df.to_excel(writer, sheet_name='Best Full Parameters', index=False)
        
        # 8. 特征信息
        feature_info = pd.DataFrame({
            'Feature': feature_cols,
            'Description': ['Molecular descriptor'] * len(feature_cols)
        })
        feature_info.to_excel(writer, sheet_name='Feature Information', index=False)
        
        # 9. 参数调优范围与默认值对比
        param_comparison_data = []
        for param, values in results['param_grid'].items():
            param_comparison_data.append({
                'Parameter': param,
                'Tuning Range': str(values),
                'Default Value': results['default_params'].get(param, 'N/A'),
                'Best Value': results['best_params'].get(param, 'N/A')
            })
        
        param_comparison_df = pd.DataFrame(param_comparison_data)
        param_comparison_df.to_excel(writer, sheet_name='Parameter Comparison', index=False)

def main():
    try:
        input_file = "CaCo-2_molecular_descriptors_reduced.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info(f"开始处理文件: {input_file}")
        start_time = time.time()
        
        # 加载和准备数据
        X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(input_file)
        
        # 优化SVR模型
        svr_model, svr_search = optimize_svr_model(X_train, y_train, feature_cols)
        
        # 评估训练集性能
        y_train_pred = svr_model.predict(X_train)
        train_metrics = evaluate_model(y_train, y_train_pred)
        
        # 评估测试集性能
        y_test_pred = svr_model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_test_pred)
        
        # 记录结果
        logger.info("\n=== SVR模型性能 ===")
        logger.info("训练集性能:")
        logger.info(f"MSE: {train_metrics['MSE']:.4f}")
        logger.info(f"RMSE: {train_metrics['RMSE']:.4f}")
        logger.info(f"MAE: {train_metrics['MAE']:.4f}")
        logger.info(f"R2: {train_metrics['R2']:.4f}")
        logger.info(f"Explained Variance: {train_metrics['Explained Variance']:.4f}")
        logger.info(f"MRE (%): {train_metrics['MRE (%)']:.4f}")
        
        logger.info("\n测试集性能:")
        logger.info(f"MSE: {test_metrics['MSE']:.4f}")
        logger.info(f"RMSE: {test_metrics['RMSE']:.4f}")
        logger.info(f"MAE: {test_metrics['MAE']:.4f}")
        logger.info(f"R2: {test_metrics['R2']:.4f}")
        logger.info(f"Explained Variance: {test_metrics['Explained Variance']:.4f}")
        logger.info(f"MRE (%): {test_metrics['MRE (%)']:.4f}")
        
        # 绘制散点图和残差图（包含默认参数对比）
        default_metrics = plot_scatter(y_train, y_train_pred, y_test, y_test_pred, "SVR_Optimized", 
                                      X_train, X_test, y_train, y_test, feature_cols)
        
        # 保存散点图数据到Excel
        save_scatter_data(y_train, y_train_pred, y_test, y_test_pred, X_train, X_test, 
                         y_train, y_test, feature_cols, "SVR_Optimized")
        
        # 绘制适用域分析图
        plot_applicability_domain(svr_model, X_train, X_test, y_train, y_test, "SVR_Optimized")
        
        # 记录默认参数模型的性能
        logger.info("\n=== 默认参数SVR模型性能 ===")
        logger.info("训练集性能:")
        logger.info(f"MSE: {default_metrics['train']['MSE']:.4f}")
        logger.info(f"RMSE: {default_metrics['train']['RMSE']:.4f}")
        logger.info(f"MAE: {default_metrics['train']['MAE']:.4f}")
        logger.info(f"R2: {default_metrics['train']['R2']:.4f}")
        
        logger.info("\n测试集性能:")
        logger.info(f"MSE: {default_metrics['test']['MSE']:.4f}")
        logger.info(f"RMSE: {default_metrics['test']['RMSE']:.4f}")
        logger.info(f"MAE: {default_metrics['test']['MAE']:.4f}")
        logger.info(f"R2: {default_metrics['test']['R2']:.4f}")
        
        # 绘制残差图
        plot_residuals(y_train, y_train_pred, y_test, y_test_pred, "SVR_Optimized")

        # 绘制残差直方图（仅调优模型测试集）
        tuned_test_residuals = (y_test - y_test_pred).to_numpy()
        plot_residual_histogram(tuned_test_residuals, "svr_tuned_test", "svr_results")
        
        # 只创建SHAP分析图表，不创建雷达图和其他图表
        feature_importance, shap_values = create_shap_analysis(
            X_train, X_test, y_train, y_test, feature_cols, "SVR_Optimized"
        )
        
        # 記錄特徵重要性
        logger.info("\n=== 特徵重要性分析（基於SHAP） ===")
        logger.info("前10個最重要特徵:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            logger.info(f"{i+1}. {row['Feature']}: {row['Mean_ABS_SHAP']:.4f}")
        
        # 创建results字典用于保存到Excel
        results = {
            'default': default_metrics['test'],  # 使用测试集性能作为默认模型性能
            'tuned': test_metrics,
            'best_params': svr_search.best_params_,
            'param_grid': {
                'svr__kernel': ['rbf'],
                'svr__C': [0.5, 1.0, 1.5, 2.0],
                'svr__gamma': ['scale', 0.05, 0.1],
                'svr__epsilon': [0.05, 0.1, 0.15, 0.2]
            },
            'default_params': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            },
            'best_full_params': svr_search.best_params_
        }
        
        # 保存结果到Excel - 传入训练集和测试集指标
        save_results_to_excel(results, feature_cols, X_train, y_train, train_metrics, test_metrics)
        
        # 保存模型
        os.makedirs('saved_models', exist_ok=True)
        joblib.dump(svr_model, 'saved_models/svr_optimized_model.pkl')
        
        logger.info("模型已保存至 saved_models/ 目录")
        
        # 执行时间
        elapsed = time.time() - start_time
        logger.info(f"\n总执行时间: {elapsed:.2f}秒")
        logger.info("处理完成!")
        
        return train_metrics, test_metrics, svr_model, svr_search, default_metrics
    
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise

if __name__ == "__main__":
    train_results, test_results, svr_model, svr_search, default_results = main()