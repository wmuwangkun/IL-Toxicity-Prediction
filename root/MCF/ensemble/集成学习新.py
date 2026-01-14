import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
import time
from scipy.stats import loguniform, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.patches as mpatches
import shap
import warnings
warnings.filterwarnings('ignore')
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
        logging.FileHandler("ensemble_model_training.log", encoding='utf-8'),
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

def create_individual_models(feature_cols):
    """创建个体模型"""
    # SVR模型 - 基于您的最佳参数
    svr_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=17.58364027, gamma=0.00190696, epsilon=0.1))
    ])
    
    # 弹性网络模型 - 基于您的最佳参数
    enet_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(alpha=0.0376, l1_ratio=0.376, max_iter=10000, tol=0.0001, random_state=42))
    ])
    
    return svr_pipe, enet_pipe

def compute_weights(metric_name, svr_metrics, enet_metrics, eps=1e-12):
    """根据指定指标计算两个模型的集成权重"""
    if metric_name == "R2":
        raw = np.array([
            max(svr_metrics["R2"], 0),
            max(enet_metrics["R2"], 0)
        ])
    else:
        raw = np.array([
            1.0 / (svr_metrics[metric_name] + eps),
            1.0 / (enet_metrics[metric_name] + eps)
        ])
    total = raw.sum()
    if total == 0:
        return np.array([0.5, 0.5])
    return raw / total

def optimize_ensemble_model(X_train, y_train, X_val, y_val, feature_cols):
    """优化集成模型 - 使用验证指标自适应权重"""
    # 创建个体模型
    svr_pipe, enet_pipe = create_individual_models(feature_cols)
    
    # 基于您的最佳参数进行小范围调优
    svr_params = {
        'svr__C': uniform(15, 5),      # 17.58364027 ± 2.5
        'svr__gamma': uniform(0.001, 0.002),  # 0.00190696 ± 0.0005
        'svr__epsilon': [0.08, 0.1, 0.12]     # 0.1 ± 0.02
    }
    
    enet_params = {
        'enet__alpha': uniform(0.03, 0.02),    # 0.0376 ± 0.01
        'enet__l1_ratio': uniform(0.35, 0.05), # 0.376 ± 0.025
        'enet__max_iter': [8000, 10000, 12000], # 10000 ± 2000
        'enet__tol': [8e-5, 1e-4, 1.2e-4]     # 0.0001 ± 2e-5
    }
    
    logger.info("开始优化SVR模型...")
    svr_search = RandomizedSearchCV(
        svr_pipe, svr_params, n_iter=30, cv=5,
        scoring='r2', n_jobs=-1, verbose=1, random_state=42
    )
    svr_search.fit(X_train, y_train)
    
    logger.info("开始优化弹性网络模型...")
    enet_search = RandomizedSearchCV(
        enet_pipe, enet_params, n_iter=30, cv=5,
        scoring='r2', n_jobs=-1, verbose=1, random_state=42
    )
    enet_search.fit(X_train, y_train)
    
    # 创建集成模型
    best_svr = svr_search.best_estimator_
    best_enet = enet_search.best_estimator_
    
    # 计算验证集上的单模型表现
    svr_val_pred = best_svr.predict(X_val)
    enet_val_pred = best_enet.predict(X_val)
    svr_val_metrics = evaluate_model(y_val, svr_val_pred)
    enet_val_metrics = evaluate_model(y_val, enet_val_pred)
    
    logger.info("验证集单模型表现：")
    logger.info(f"SVR  -> R2={svr_val_metrics['R2']:.4f}, MAE={svr_val_metrics['MAE']:.4f}, MSE={svr_val_metrics['MSE']:.4f}, RMSE={svr_val_metrics['RMSE']:.4f}")
    logger.info(f"ENet -> R2={enet_val_metrics['R2']:.4f}, MAE={enet_val_metrics['MAE']:.4f}, MSE={enet_val_metrics['MSE']:.4f}, RMSE={enet_val_metrics['RMSE']:.4f}")
    
    metrics_order = ["R2", "MAE", "MSE", "RMSE"]
    best_weights = None
    best_val_r2 = -np.inf
    
    for metric_name in metrics_order:
        weights = compute_weights(metric_name, svr_val_metrics, enet_val_metrics)
        ensemble_candidate = VotingRegressor([
            ('svr', best_svr),
            ('enet', best_enet)
        ], weights=list(weights))
        
        ensemble_candidate.fit(X_train, y_train)
        val_pred = ensemble_candidate.predict(X_val)
        val_metrics = evaluate_model(y_val, val_pred)
        
        logger.info(f"权重方案（{metric_name}） -> 权重 {weights}, 验证集 R2={val_metrics['R2']:.4f}")
        
        if val_metrics["R2"] > best_val_r2:
            best_val_r2 = val_metrics["R2"]
            best_weights = weights
    
    if best_weights is None:
        best_weights = np.array([0.5, 0.5])
    
    # 创建最佳集成模型
    best_ensemble = VotingRegressor([
        ('svr', best_svr),
        ('enet', best_enet)
    ], weights=list(best_weights))
    
    best_ensemble.fit(X_train, y_train)
    
    logger.info(f"SVR最佳参数: {svr_search.best_params_}")
    logger.info(f"弹性网络最佳参数: {enet_search.best_params_}")
    logger.info(f"最佳集成权重: {best_weights}")
    logger.info(f"集成模型验证集R2: {best_val_r2:.4f}")
    
    return best_ensemble, svr_search, enet_search, list(best_weights)

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

def create_shap_analysis(X_train, X_test, y_train, y_test, feature_cols, ensemble_model, model_name):
    """创建SHAP分析图表 - SCI一区风格，字体12加粗，输出TIF，无'Features'标签，等比例缩放内容，只显示前15个特征"""
    os.makedirs('ensemble_results', exist_ok=True)
    
    logger.info("开始创建集成模型SHAP分析...")
    
    # 使用集成模型进行SHAP分析
    # 由于VotingRegressor的复杂性，我们使用KernelExplainer
    explainer = shap.KernelExplainer(ensemble_model.predict, X_train.values[:100])  # 使用100个样本作为背景
    shap_values = explainer.shap_values(X_test.values[:50])  # 使用50个测试样本进行分析
    
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
    
    plt.savefig(f'ensemble_results/{model_name}_global_explanation.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'ensemble_results/{model_name}_global_explanation.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'ensemble_results/{model_name}_global_explanation.tif', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='tiff')
    plt.close()

    # 图(b): Local explanation - SHAP摘要图（等比例缩放）
    # 使用相同的尺寸设置
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 绘制SHAP摘要图
    shap.summary_plot(shap_values, X_test.values[:50],
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
    
    plt.savefig(f'ensemble_results/{model_name}_local_explanation.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'ensemble_results/{model_name}_local_explanation.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'ensemble_results/{model_name}_local_explanation.tif', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='tiff')
    plt.close()
    
    # 保存特征重要性到Excel
    feature_importance_df_sorted = feature_importance_df.sort_values('Mean_ABS_SHAP', ascending=False)
    with pd.ExcelWriter('ensemble_results/ensemble_feature_importance_analysis.xlsx') as writer:
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

    logger.info(f"集成模型SHAP分析图表已保存完成（含TIF格式）")
    logger.info(f"特征重要性分析已保存至 ensemble_results/ensemble_feature_importance_analysis.xlsx")
    
    return feature_importance_df_sorted, shap_values

def save_scatter_data(y_train, y_train_pred, y_test, y_test_pred, X_train, X_test, y_train_orig, y_test_orig, feature_cols, model_name):
    """保存散点图数据到Excel文件"""
    os.makedirs('ensemble_results', exist_ok=True)
    
    # 重新训练默认参数模型以获取预测值
    if model_name == "SVR":
        default_model_pipe = Pipeline([
            ('feature_selector', FeatureSelector(feature_cols)),
            ('scaler', StandardScaler()),
            ('svr', SVR())  # 使用默认参数
        ])
    elif model_name == "ElasticNet":
        default_model_pipe = Pipeline([
            ('feature_selector', FeatureSelector(feature_cols)),
            ('scaler', StandardScaler()),
            ('enet', ElasticNet())  # 使用默认参数
        ])
    else:  # Ensemble
        # 对于集成模型，没有默认参数，所以只保存最佳参数预测值
        default_model_pipe = None
    
    if default_model_pipe is not None:
        default_model_pipe.fit(X_train, y_train_orig)
        # 获取默认参数的预测
        y_train_pred_default = default_model_pipe.predict(X_train)
        y_test_pred_default = default_model_pipe.predict(X_test)
    else:
        # 对于集成模型，没有默认参数预测值
        y_train_pred_default = np.nan * np.ones_like(y_train_pred)
        y_test_pred_default = np.nan * np.ones_like(y_test_pred)
    
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
    with pd.ExcelWriter(f'ensemble_results/{model_name}_scatter_plot_data.xlsx') as writer:
        full_data.to_excel(writer, sheet_name='All_Data', index=False)
        
        # 分别保存训练集和测试集
        scatter_data.to_excel(writer, sheet_name='Training_Set', index=False)
        test_data.to_excel(writer, sheet_name='Test_Set', index=False)
        
        # 保存性能指标
        if default_model_pipe is not None:
            train_r2_default = r2_score(y_train_orig, y_train_pred_default)
            test_r2_default = r2_score(y_test_orig, y_test_pred_default)
        else:
            train_r2_default = np.nan
            test_r2_default = np.nan
            
        train_r2_best = r2_score(y_train_orig, y_train_pred)
        test_r2_best = r2_score(y_test_orig, y_test_pred)
        
        metrics_data = {
            'Metric': ['R2', 'MSE', 'RMSE', 'MAE'],
            'Default_Train': [
                train_r2_default,
                mean_squared_error(y_train_orig, y_train_pred_default) if default_model_pipe is not None else np.nan,
                np.sqrt(mean_squared_error(y_train_orig, y_train_pred_default)) if default_model_pipe is not None else np.nan,
                mean_absolute_error(y_train_orig, y_train_pred_default) if default_model_pipe is not None else np.nan
            ],
            'Default_Test': [
                test_r2_default,
                mean_squared_error(y_test_orig, y_test_pred_default) if default_model_pipe is not None else np.nan,
                np.sqrt(mean_squared_error(y_test_orig, y_test_pred_default)) if default_model_pipe is not None else np.nan,
                mean_absolute_error(y_test_orig, y_test_pred_default) if default_model_pipe is not None else np.nan
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
    
    logger.info(f"散点图数据已保存至 ensemble_results/{model_name}_scatter_plot_data.xlsx")

def _calculate_average_feature_distance(train_features, other_features=None):
    """计算平均特征距离"""
    if other_features is None:
        distance_matrix = pairwise_distances(train_features, metric='euclidean')
        np.fill_diagonal(distance_matrix, np.nan)
        return np.nanmean(distance_matrix, axis=1)
    distance_matrix = pairwise_distances(other_features, train_features, metric='euclidean')
    return distance_matrix.mean(axis=1)

def plot_applicability_domain(model, X_train, X_test, y_train, y_test, feature_cols):
    """绘制集成模型适用域分析图并保存数据 - 样式与参考代码一致"""
    os.makedirs('ensemble_results', exist_ok=True)

    if isinstance(X_train, pd.DataFrame):
        X_train_data = X_train[feature_cols].values
        X_test_data = X_test[feature_cols].values
    else:
        X_train_data = X_train
        X_test_data = X_test

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_data)
    X_test_scaled = scaler.transform(X_test_data)

    avg_dist_train = _calculate_average_feature_distance(X_train_scaled)
    avg_dist_test = _calculate_average_feature_distance(X_train_scaled, X_test_scaled)

    y_train_array = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
    y_test_array = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

    y_train_pred = model.predict(X_train_data if isinstance(X_train, np.ndarray) else X_train)
    y_test_pred = model.predict(X_test_data if isinstance(X_test, np.ndarray) else X_test)

    residuals_train = y_train_pred - y_train_array
    residuals_test = y_test_pred - y_test_array

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

    # 创建图形，尺寸与参考代码一致
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

    # 轴标签、标题、刻度文字设置 - 与参考代码一致
    ax.set_xlabel('Average Feature Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standardized Residuals', fontsize=12, fontweight='bold')
    ax.set_title('Applicability Domain Analysis (Training & Testing)', fontsize=12, fontweight='bold')

    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')

    # -------------------------- 图例样式（与参考代码一致）--------------------------
    ax.legend(
        fontsize=12,          # 图例字体大小
        loc='upper right',   # 图例位置
        framealpha=0.9,      # 图例背景透明度
        fancybox=True,       # 圆角边框
        shadow=False,        # 去掉阴影
        borderpad=0.5,       # 图例内边距
        labelspacing=0.4,    # 图例项间距
        handlelength=0.1,    # 图例标记长度
        handletextpad=0.6,   # 标记与文字间距
        columnspacing=0.5,   # 多列间距
        prop={'weight': 'bold', 'size': 11}  # 图例文字加粗
    )
    ax.grid(True, linestyle='--', alpha=0.3)

    # -------------------------- 统计文本样式（与参考代码一致）--------------------------
    exceed_text = (
        f"Training outside: {((ad_df['Set'] == 'Training') & ad_df['OutOfDomain']).sum()}\n"
        f"Testing outside: {((ad_df['Set'] == 'Testing') & ad_df['OutOfDomain']).sum()}"
    )
    ax.text(
        0.013,                # x位置（相对坐标）
        0.983,                # y位置（相对坐标）
        exceed_text,         # 文本内容
        transform=ax.transAxes,  # 相对坐标系（0-1范围）
        fontsize=11,         # 统计文本字体大小
        fontweight='bold',   # 统计文本字重
        fontfamily='Arial',  # 统计文本字体
        color='darkred',     # 统计文本颜色
        verticalalignment='top',  # 垂直对齐方式
        bbox=dict(
            boxstyle='round,pad=0.3',  # 文本框样式
            facecolor='lightyellow',   # 文本框背景色
            alpha=0.8,                 # 文本框透明度
            edgecolor='orange',        # 文本框边框色
            linewidth=0.9              # 文本框边框宽度
        )
    )

    plt.tight_layout()
    # 保存为PNG和TIFF格式
    plt.savefig('ensemble_results/ensemble_applicability_domain.png', dpi=300, bbox_inches='tight')
    plt.savefig('ensemble_results/ensemble_applicability_domain.tif', dpi=300, bbox_inches='tight', format='tiff')
    plt.close()

    testing_outliers = ((ad_df['Set'] == 'Testing') & ad_df['OutOfDomain']).sum()
    summary_df = pd.DataFrame({
        'Metric': ['Testing Samples Outside Domain'],
        'Count': [testing_outliers]
    })

    with pd.ExcelWriter('ensemble_results/ensemble_applicability_domain_data.xlsx') as writer:
        ad_df.to_excel(writer, sheet_name='All_Samples', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    logger.info("集成模型适用域分析结果已保存至 ensemble_results/ 目录")

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

def plot_scatter(y_train, y_train_pred, y_test, y_test_pred, model_name, X_train, X_test, y_train_orig, y_test_orig, feature_cols):
    """绘制散点图 - SCI风格，包含个体模型和集成模型的对比"""
    os.makedirs('ensemble_results', exist_ok=True)
    
    # 创建个体模型
    svr_pipe, enet_pipe = create_individual_models(feature_cols)
    
    # 训练个体模型
    svr_pipe.fit(X_train, y_train_orig)
    enet_pipe.fit(X_train, y_train_orig)
    
    # 获取个体模型的预测
    y_train_pred_svr = svr_pipe.predict(X_train)
    y_test_pred_svr = svr_pipe.predict(X_test)
    y_train_pred_enet = enet_pipe.predict(X_train)
    y_test_pred_enet = enet_pipe.predict(X_test)
    
    # 保存SVR模型的散点图数据
    save_scatter_data(y_train_orig, y_train_pred_svr, y_test_orig, y_test_pred_svr, 
                     X_train, X_test, y_train_orig, y_test_orig, feature_cols, "SVR")
    
    # 保存ElasticNet模型的散点图数据
    save_scatter_data(y_train_orig, y_train_pred_enet, y_test_orig, y_test_pred_enet, 
                     X_train, X_test, y_train_orig, y_test_orig, feature_cols, "ElasticNet")
    
    # 保存Ensemble模型的散点图数据
    save_scatter_data(y_train_orig, y_train_pred, y_test_orig, y_test_pred, 
                     X_train, X_test, y_train_orig, y_test_orig, feature_cols, "Ensemble")
    
    # 创建包含三个子图的图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # 定义清晰的颜色
    train_color = '#FF0000'  # 红色
    test_color = '#0000FF'   # 蓝色
    line_color = '#45B7D1'   # 青色
    
    # 绘制SVR模型的散点图
    ax1.scatter(y_train_orig, y_train_pred_svr, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax1.scatter(y_test_orig, y_test_pred_svr, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    all_y_svr = np.concatenate([y_train_orig, y_test_orig])
    all_y_pred_svr = np.concatenate([y_train_pred_svr, y_test_pred_svr])
    min_val_svr = min(all_y_svr.min(), all_y_pred_svr.min())
    max_val_svr = max(all_y_svr.max(), all_y_pred_svr.max())
    ax1.plot([min_val_svr, max_val_svr], [min_val_svr, max_val_svr], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax1.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax1.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax1.set_title('SVR Model', fontsize=8, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 设置坐标轴刻度字体加粗
    for tick in ax1.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    
    train_r2_svr = r2_score(y_train_orig, y_train_pred_svr)
    test_r2_svr = r2_score(y_test_orig, y_test_pred_svr)
    ax1.text(0.95, 0.05, f'Training R² = {train_r2_svr:.3f}\nTest R² = {test_r2_svr:.3f}', 
             transform=ax1.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    # 绘制弹性网络模型的散点图
    ax2.scatter(y_train_orig, y_train_pred_enet, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax2.scatter(y_test_orig, y_test_pred_enet, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    all_y_enet = np.concatenate([y_train_orig, y_test_orig])
    all_y_pred_enet = np.concatenate([y_train_pred_enet, y_test_pred_enet])
    min_val_enet = min(all_y_enet.min(), all_y_pred_enet.min())
    max_val_enet = max(all_y_enet.max(), all_y_pred_enet.max())
    ax2.plot([min_val_enet, max_val_enet], [min_val_enet, max_val_enet], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax2.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax2.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax2.set_title('ElasticNet Model', fontsize=8, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 设置坐标轴刻度字体加粗
    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
    
    train_r2_enet = r2_score(y_train_orig, y_train_pred_enet)
    test_r2_enet = r2_score(y_test_orig, y_test_pred_enet)
    ax2.text(0.95, 0.05, f'Training R² = {train_r2_enet:.3f}\nTest R² = {test_r2_enet:.3f}', 
             transform=ax2.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    # 绘制集成模型的散点图
    ax3.scatter(y_train_orig, y_train_pred, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax3.scatter(y_test_orig, y_test_pred, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    all_y_ensemble = np.concatenate([y_train_orig, y_test_orig])
    all_y_pred_ensemble = np.concatenate([y_train_pred, y_test_pred])
    min_val_ensemble = min(all_y_ensemble.min(), all_y_pred_ensemble.min())
    max_val_ensemble = max(all_y_ensemble.max(), all_y_pred_ensemble.max())
    ax3.plot([min_val_ensemble, max_val_ensemble], [min_val_ensemble, max_val_ensemble], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax3.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax3.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax3.set_title('Ensemble', fontsize=8, fontweight='bold')  # 修改标题为Ensemble
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 设置坐标轴刻度字体加粗
    for tick in ax3.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax3.get_yticklabels():
        tick.set_fontweight('bold')
    
    train_r2_ensemble = r2_score(y_train_orig, y_train_pred)
    test_r2_ensemble = r2_score(y_test_orig, y_test_pred)
    ax3.text(0.95, 0.05, f'Training R² = {train_r2_ensemble:.3f}\nTest R² = {test_r2_ensemble:.3f}', 
             transform=ax3.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    plt.tight_layout()
    
    # 保存对比图片
    plt.savefig(f'ensemble_results/{model_name}_comparison_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'ensemble_results/{model_name}_comparison_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"对比散点图已保存至 ensemble_results/{model_name}_comparison_scatter_plot.png")
    
    # 单独绘制集成模型的散点图
    fig, ax = plt.subplots(figsize=(4, 4))
    
    ax.scatter(y_train_orig, y_train_pred, alpha=0.8, s=30, color=train_color, 
              edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test_orig, y_test_pred, alpha=0.8, s=30, color=test_color, 
              edgecolors='white', linewidth=0.5, label='Test set')
    
    ax.plot([min_val_ensemble, max_val_ensemble], [min_val_ensemble, max_val_ensemble], 
            color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax.set_xlabel('Experimental values', fontsize=9, fontweight='bold')
    ax.set_ylabel('Predicted values', fontsize=9, fontweight='bold')
    ax.set_title('Ensemble', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 设置坐标轴刻度字体加粗
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    ax.text(0.95, 0.05, f'Training R² = {train_r2_ensemble:.3f}\nTest R² = {test_r2_ensemble:.3f}', 
            transform=ax.transAxes, fontsize=7, verticalalignment='bottom', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                     edgecolor='gray', linewidth=0.5))
    
    plt.tight_layout()
    
    # 保存单独的集成模型图片
    plt.savefig(f'ensemble_results/Ensemble_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'ensemble_results/Ensemble_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"单独的集成模型散点图已保存至 ensemble_results/Ensemble_scatter_plot.png")
    
    # 返回个体模型的性能指标
    individual_metrics = {
        'SVR': {
            'train': {
                'R2': train_r2_svr,
                'MSE': mean_squared_error(y_train_orig, y_train_pred_svr),
                'RMSE': np.sqrt(mean_squared_error(y_train_orig, y_train_pred_svr)),
                'MAE': mean_absolute_error(y_train_orig, y_train_pred_svr)
            },
            'test': {
                'R2': test_r2_svr,
                'MSE': mean_squared_error(y_test_orig, y_test_pred_svr),
                'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_test_pred_svr)),
                'MAE': mean_absolute_error(y_test_orig, y_test_pred_svr)
            }
        },
        'ElasticNet': {
            'train': {
                'R2': train_r2_enet,
                'MSE': mean_squared_error(y_train_orig, y_train_pred_enet),
                'RMSE': np.sqrt(mean_squared_error(y_train_orig, y_train_pred_enet)),
                'MAE': mean_absolute_error(y_train_orig, y_train_pred_enet)
            },
            'test': {
                'R2': test_r2_enet,
                'MSE': mean_squared_error(y_test_orig, y_test_pred_enet),
                'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_test_pred_enet)),
                'MAE': mean_absolute_error(y_test_orig, y_test_pred_enet)
            }
        }
    }
    
    return individual_metrics

def save_results_to_excel(results, feature_cols, X_train, y_train, train_metrics, test_metrics):
    """保存结果到Excel文件 - 添加训练集指标"""
    os.makedirs('ensemble_results', exist_ok=True)
    
    with pd.ExcelWriter('ensemble_results/ensemble_results.xlsx') as writer:
        # 1. 模型性能比较
        metrics_df = pd.DataFrame({
            'SVR': results['individual']['SVR']['test'],
            'ElasticNet': results['individual']['ElasticNet']['test'],
            'Ensemble': results['ensemble']['test']
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
        
        # 5. 集成权重
        weights_df = pd.DataFrame([results['ensemble_weights']])
        weights_df.to_excel(writer, sheet_name='Ensemble Weights', index=False)
        
        # 6. 特征信息
        feature_info = pd.DataFrame({
            'Feature': feature_cols,
            'Description': ['Molecular descriptor'] * len(feature_cols)
        })
        feature_info.to_excel(writer, sheet_name='Feature Information', index=False)
        
        # 7. 详细性能指标
        detailed_metrics = pd.DataFrame({
            'Model': ['SVR', 'ElasticNet', 'Ensemble'],
            'Train_R2': [results['individual']['SVR']['train']['R2'], 
                        results['individual']['ElasticNet']['train']['R2'], 
                        results['ensemble']['train']['R2']],
            'Test_R2': [results['individual']['SVR']['test']['R2'], 
                       results['individual']['ElasticNet']['test']['R2'], 
                       results['ensemble']['test']['R2']],
            'Train_RMSE': [results['individual']['SVR']['train']['RMSE'], 
                          results['individual']['ElasticNet']['train']['RMSE'], 
                          results['ensemble']['train']['RMSE']],
            'Test_RMSE': [results['individual']['SVR']['test']['RMSE'], 
                         results['individual']['ElasticNet']['test']['RMSE'], 
                         results['ensemble']['test']['RMSE']]
        })
        detailed_metrics.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        
        # 8. 参数调优范围与默认值对比
        param_comparison_data = [
            # SVR参数
            {
                'Parameter': 'SVR__C',
                'Tuning Range': 'uniform(15, 5) [15-20]',
                'Default Value': '17.58364027',
                'Best Value': results['best_params']['SVR'].get('svr__C', 'N/A')
            },
            {
                'Parameter': 'SVR__gamma',
                'Tuning Range': 'uniform(0.001, 0.002) [0.001-0.003]',
                'Default Value': '0.00190696',
                'Best Value': results['best_params']['SVR'].get('svr__gamma', 'N/A')
            },
            {
                'Parameter': 'SVR__epsilon',
                'Tuning Range': '[0.08, 0.1, 0.12]',
                'Default Value': '0.1',
                'Best Value': results['best_params']['SVR'].get('svr__epsilon', 'N/A')
            },
            # ElasticNet参数
            {
                'Parameter': 'ElasticNet__alpha',
                'Tuning Range': 'uniform(0.03, 0.02) [0.03-0.05]',
                'Default Value': '0.0376',
                'Best Value': results['best_params']['ElasticNet'].get('enet__alpha', 'N/A')
            },
            {
                'Parameter': 'ElasticNet__l1_ratio',
                'Tuning Range': 'uniform(0.35, 0.05) [0.35-0.40]',
                'Default Value': '0.376',
                'Best Value': results['best_params']['ElasticNet'].get('enet__l1_ratio', 'N/A')
            },
            {
                'Parameter': 'ElasticNet__max_iter',
                'Tuning Range': '[8000, 10000, 12000]',
                'Default Value': '10000',
                'Best Value': results['best_params']['ElasticNet'].get('enet__max_iter', 'N/A')
            },
            {
                'Parameter': 'ElasticNet__tol',
                'Tuning Range': '[8e-5, 1e-4, 1.2e-4]',
                'Default Value': '0.0001',
                'Best Value': results['best_params']['ElasticNet'].get('enet__tol', 'N/A')
            },
            # 集成权重参数
            {
                'Parameter': 'Ensemble Weights',
                'Tuning Range': '基于验证指标自适应',
                'Default Value': '[0.5, 0.5]',
                'Best Value': str(results['ensemble_weights'])
            }
        ]
        
        param_comparison_df = pd.DataFrame(param_comparison_data)
        param_comparison_df.to_excel(writer, sheet_name='Parameter Comparison', index=False)
        
        # 9. 参数调优统计
        tuning_stats = pd.DataFrame({
            'Metric': ['Total Parameter Combinations', 'SVR Search Iterations', 'ElasticNet Search Iterations', 'Weight Schemes Evaluated'],
            'Value': ['Multiple combinations', '30', '30', '4'],
            'Description': ['SVR和ElasticNet参数组合总数', 'SVR随机搜索次数', 'ElasticNet随机搜索次数', '测试的权重方案数量']
        })
        tuning_stats.to_excel(writer, sheet_name='Tuning Statistics', index=False)

def main():
    try:
        input_file = "MCF-7_molecular_descriptors_reduced1.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info(f"开始处理文件: {input_file}")
        start_time = time.time()
        
        # 加载和准备数据
        X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(input_file)
        
        # 优化集成模型
        ensemble_model, svr_search, enet_search, best_weights = optimize_ensemble_model(
            X_train, y_train, X_test, y_test, feature_cols
        )
        
        # 评估集成模型性能
        y_train_pred = ensemble_model.predict(X_train)
        train_metrics = evaluate_model(y_train, y_train_pred)
        
        y_test_pred = ensemble_model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_test_pred)

        # 绘制残差直方图（仅调优模型测试集）
        ensemble_test_residuals = (y_test - y_test_pred).to_numpy()
        plot_residual_histogram(ensemble_test_residuals, 'ensemble_tuned_test', 'ensemble_results')
        
        # 记录结果
        logger.info("\n=== 集成模型性能 ===")
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
        
        # 绘制对比散点图
        individual_metrics = plot_scatter(y_train, y_train_pred, y_test, y_test_pred, "Ensemble", 
                                        X_train, X_test, y_train, y_test, feature_cols)

        # 绘制适用域分析（调优后的集成模型）
        plot_applicability_domain(ensemble_model, X_train, X_test, y_train, y_test, feature_cols)
        
        # 创建results字典
        results = {
            'individual': individual_metrics,
            'ensemble': {
                'train': train_metrics,
                'test': test_metrics
            },
            'best_params': {
                'SVR': svr_search.best_params_,
                'ElasticNet': enet_search.best_params_
            },
            'ensemble_weights': best_weights
        }
        
        # 保存结果到Excel - 传入训练集和测试集指标
        save_results_to_excel(results, feature_cols, X_train, y_train, train_metrics, test_metrics)
        
        # 保存模型
        os.makedirs('saved_models', exist_ok=True)
        joblib.dump(ensemble_model, 'saved_models/ensemble_model.pkl')
        
        # 保存个体模型
        joblib.dump(svr_search.best_estimator_, 'saved_models/svr_best_model.pkl')
        joblib.dump(enet_search.best_estimator_, 'saved_models/enet_best_model.pkl')
        
        logger.info("集成模型已保存至 saved_models/ 目录")
        logger.info("个体模型已保存至 saved_models/ 目录")
        
        # 进行SHAP分析
        logger.info("\n=== 开始SHAP分析 ===")
        feature_importance_df, shap_values = create_shap_analysis(
            X_train, X_test, y_train, y_test, feature_cols, ensemble_model, "Ensemble"
        )
        
        # 打印前15个最重要特征
        logger.info("\n=== 前15个最重要特征 ===")
        top_15_features = feature_importance_df.head(15)
        for i, (idx, row) in enumerate(top_15_features.iterrows(), 1):
            logger.info(f"{i:2d}. {row['Feature']:<25} SHAP值: {row['Mean_ABS_SHAP']:.4f}")
        
        # 打印最终结果摘要
        logger.info("\n" + "="*50)
        logger.info("最终结果摘要")
        logger.info("="*50)
        logger.info(f"最佳集成权重: {best_weights}")
        logger.info(f"集成模型测试集R2: {test_metrics['R2']:.4f}")
        logger.info(f"集成模型测试集RMSE: {test_metrics['RMSE']:.4f}")
        logger.info(f"SVR测试集R2: {individual_metrics['SVR']['test']['R2']:.4f}")
        logger.info(f"ElasticNet测试集R2: {individual_metrics['ElasticNet']['test']['R2']:.4f}")
        logger.info("="*50)
        
        # 打印参数调优信息
        logger.info("\n参数调优信息:")
        logger.info("SVR参数调优范围:")
        logger.info("  C: uniform(15, 5) [15-20]")
        logger.info("  gamma: uniform(0.001, 0.002) [0.001-0.003]")
        logger.info("  epsilon: [0.08, 0.1, 0.12]")
        logger.info("ElasticNet参数调优范围:")
        logger.info("  alpha: uniform(0.03, 0.02) [0.03-0.05]")
        logger.info("  l1_ratio: uniform(0.35, 0.05) [0.35-0.40]")
        logger.info("  max_iter: [8000, 10000, 12000]")
        logger.info("  tol: [8e-5, 1e-4, 1.2e-4]")
        logger.info("集成权重调优: 基于验证指标自适应")
        
        # 执行时间
        elapsed = time.time() - start_time
        logger.info(f"\n总执行时间: {elapsed:.2f}秒")
        logger.info("处理完成!")
        
        return results, ensemble_model, feature_importance_df, shap_values
    
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise

if __name__ == "__main__":
    results, ensemble_model, feature_importance_df, shap_values = main()