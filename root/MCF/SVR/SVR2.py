import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
import time
from scipy.stats import loguniform
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.patches as mpatches

# 设置matplotlib参数，适合SCI发表
plt.rcParams.update({
    'font.size': 6,           # 基础字体大小
    'axes.titlesize': 7,      # 标题字体大小
    'axes.labelsize': 6,      # 轴标签字体大小
    'xtick.labelsize': 5,     # x轴刻度标签字体大小
    'ytick.labelsize': 5,     # y轴刻度标签字体大小
    'legend.fontsize': 5,     # 图例字体大小
    'figure.titlesize': 8,    # 图形标题字体大小
    'lines.linewidth': 1.2,   # 线条宽度
    'axes.linewidth': 0.8,    # 轴线宽度
    'grid.linewidth': 0.4,    # 网格线宽度
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
    
    svr_params = {
        'svr__kernel': ['rbf', 'linear'],
        'svr__C': loguniform(1e-1, 1e3),
        'svr__gamma': loguniform(1e-4, 1e1),
        'svr__epsilon': [0.01, 0.05, 0.1, 0.5]
    }
    
    logger.info("开始优化SVR模型...")
    svr_search = RandomizedSearchCV(
        svr_pipe, svr_params, n_iter=50, cv=5,
        scoring='r2', n_jobs=-1, verbose=1, random_state=42
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
    
    # 设置坐标轴刻度标签加粗 - 方法四：更粗的字体
    for label in ax1.get_xticklabels():
        label.set_fontweight('heavy')  # 改為heavy
        label.set_fontsize(7)          # 改為7
        label.set_color('#000000')     # 純黑色，更突出
    for label in ax1.get_yticklabels():
        label.set_fontweight('heavy')  # 改為heavy
        label.set_fontsize(7)          # 改為7
        label.set_color('#000000')     # 純黑色，更突出
    
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
    
    # 设置坐标轴刻度标签加粗 - 方法四：更粗的字体
    for label in ax2.get_xticklabels():
        label.set_fontweight('heavy')  # 改為heavy
        label.set_fontsize(7)          # 改為7
        label.set_color('#000000')     # 純黑色，更突出
    for label in ax2.get_yticklabels():
        label.set_fontweight('heavy')  # 改為heavy
        label.set_fontsize(7)          # 改為7
        label.set_color('#000000')     # 純黑色，更突出
    
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
    
    # 设置坐标轴刻度标签加粗 - 方法四：更粗的字体
    for label in ax.get_xticklabels():
        label.set_fontweight('heavy')  # 改為heavy
        label.set_fontsize(7)          # 改為7
        label.set_color('#000000')     # 純黑色，更突出
    for label in ax.get_yticklabels():
        label.set_fontweight('heavy')  # 改為heavy
        label.set_fontsize(7)          # 改為7
        label.set_color('#000000')     # 純黑色，更突出
    
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


import shap
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def create_shap_analysis(X_train, X_test, y_train, y_test, feature_cols, model_name):
    """创建SHAP分析图表 - SCI一区风格，字体12加粗，输出TIF，无'Features'标签，等比例缩放内容"""
    os.makedirs('svr_results', exist_ok=True)
    
    logger.info("开始创建SHAP分析...")
    
    # 使用Random Forest进行SHAP分析
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_model.fit(X_train_scaled, y_train)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # 计算平均绝对SHAP值
    mean_shap_abs = np.mean(np.abs(shap_values), axis=0)
    
    # 创建特征重要性DataFrame并排序
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_ABS_SHAP': mean_shap_abs
    }).sort_values('Mean_ABS_SHAP', ascending=True)
    
    # 只选择最重要的15个特征用于绘图
    top_features = feature_importance_df.tail(15)  # 前15个最重要特征
    
    # 统一设置图的尺寸 - 确保两个图完全一致
    fig_width = 6
    fig_height = 8
    
    # 图(a): Global explanation - 特征重要性条形图
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 使用渐变色
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    bars = ax.barh(range(len(top_features)), 
                   top_features['Mean_ABS_SHAP'],
                   color=colors, alpha=0.85, edgecolor='none', linewidth=0)
    
    # 设置y轴标签（特征名称） - 12号加粗
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=12, fontweight='bold')
    
    # 设置x轴标签 - 12号加粗
    ax.set_xlabel('mean(|SHAP value|)', fontsize=12, fontweight='bold')
    
    # 刻度标签：x轴数字 12号加粗
    ax.tick_params(axis='x', labelsize=12, width=1.2, length=4, labelcolor='#2E2E2E')
    ax.tick_params(axis='y', width=1.2, length=4)
    
    # 手动设置x轴刻度标签加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        label.set_color('#000000')
    
    # 手动设置y轴刻度标签加粗
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        label.set_color('#000000')
    
    # 去掉上、右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 在条形上添加数值标签 - 12号加粗
    for i, (bar, value) in enumerate(zip(bars, top_features['Mean_ABS_SHAP'])):
        ax.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=12, 
                fontweight='bold', color='#2E2E2E')
    
    # 背景色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    
    # 调整子图位置，确保内容比例一致
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.1)
    
    plt.savefig(f'svr_results/{model_name}_global_explanation.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'svr_results/{model_name}_global_explanation.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'svr_results/{model_name}_global_explanation.tif', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='tiff')
    plt.close()

    # 图(b): Local explanation - SHAP摘要图（等比例缩放）
    # 使用相同的尺寸设置
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 绘制SHAP摘要图
    shap.summary_plot(shap_values, X_test_scaled,
                      feature_names=feature_cols,
                      plot_type="dot",
                      show=False,
                      max_display=15)  # 只显示top15特征
    
    ax = plt.gca()
    
    # 修改x轴标签 - 12号加粗
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')
    
    # 修改y轴标签（去掉"Features"）
    ax.set_ylabel('')  # 清空"Features"
    
    # 修改刻度标签 - 12号加粗
    ax.tick_params(axis='x', labelsize=12, width=1.2, length=4, labelcolor='#2E2E2E')
    ax.tick_params(axis='y', labelsize=12, width=1.2, length=4)
    
    # 手动设置x轴刻度标签加粗 - 强制重新设置所有x轴标签
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        label.set_color('#000000')
    
    # 手动设置y轴刻度标签加粗
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
        label.set_color('#000000')
    
    # 处理 colorbar（low/high 加粗）
    cbar = plt.gcf().axes[-1]  # 获取colorbar
    cbar.tick_params(labelsize=12)  # 数字大小
    for label in cbar.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')  # 加粗 low/high
    cbar.set_ylabel('')  # 可选：去掉colorbar标题
    
    # 重要：强制刷新图形以确保所有更改生效
    plt.draw()
    
    # 再次确保所有标签加粗（防止SHAP覆盖）
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        label.set_color('#000000')
    
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
        label.set_color('#000000')
    
    # 背景
    ax.set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    ax.grid(False)
    
    # 调整子图位置，确保内容比例与条形图一致
    plt.subplots_adjust(left=0.25, right=0.85, top=0.95, bottom=0.1)
    
    # 在保存前最后一次确保所有标签加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        label.set_color('#000000')
    
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
        label.set_color('#000000')
    
    plt.savefig(f'svr_results/{model_name}_local_explanation.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'svr_results/{model_name}_local_explanation.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'svr_results/{model_name}_local_explanation.tif', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='tiff')
    plt.close()

    # 保存特征重要性到Excel
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
        input_file = "MCF-7_molecular_descriptors_reduced1.xlsx"
        
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
                'svr__kernel': ['rbf', 'linear'],
                'svr__C': 'loguniform(1e-1, 1e3)',
                'svr__gamma': 'loguniform(1e-4, 1e1)',
                'svr__epsilon': [0.01, 0.05, 0.1, 0.5]
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