import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
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
import shap
import warnings
warnings.filterwarnings('ignore')
from matplotlib.font_manager import FontProperties
from xgboost import XGBRegressor

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
    # MLP模型 - 基于最佳参数
    mlp_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(100, 100),
            activation='relu',
            alpha=0.001,
            learning_rate_init=0.001,
            batch_size=64,
            max_iter=1000,
            random_state=42
        ))
    ])
    
    # XGBoost模型 - 基于最佳参数
    xgb_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(
            n_estimators=200,
            learning_rate=0.15,
            max_depth=6,
            subsample=0.75,
            colsample_bytree=0.85,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=0.8,
            objective='reg:squarederror',
            random_state=42
        ))
    ])
    
    return mlp_pipe, xgb_pipe

def compute_weights(metric_name, mlp_metrics, xgb_metrics, eps=1e-12):
    """根据指定指标计算两个模型的集成权重"""
    if metric_name == "R2":
        raw = np.array([
            max(mlp_metrics["R2"], 0),
            max(xgb_metrics["R2"], 0)
        ])
    else:
        raw = np.array([
            1.0 / (mlp_metrics[metric_name] + eps),
            1.0 / (xgb_metrics[metric_name] + eps)
        ])
    total = raw.sum()
    if total == 0:
        return np.array([0.5, 0.5])
    return raw / total

def optimize_ensemble_model(X_train, y_train, X_val, y_val, feature_cols):
    """构建集成模型 - 使用验证指标自适应权重"""
    # 创建个体模型
    mlp_pipe, xgb_pipe = create_individual_models(feature_cols)
    
    logger.info("开始训练MLP模型（使用最佳参数）...")
    mlp_pipe.fit(X_train, y_train)
    
    logger.info("开始训练XGBoost模型（使用最佳参数）...")
    xgb_pipe.fit(X_train, y_train)
    
    # 验证集表现
    mlp_val_pred = mlp_pipe.predict(X_val)
    xgb_val_pred = xgb_pipe.predict(X_val)
    mlp_val_metrics = evaluate_model(y_val, mlp_val_pred)
    xgb_val_metrics = evaluate_model(y_val, xgb_val_pred)
    
    logger.info("验证集单模型表现：")
    logger.info(f"MLP      -> R2={mlp_val_metrics['R2']:.4f}, MAE={mlp_val_metrics['MAE']:.4f}, MSE={mlp_val_metrics['MSE']:.4f}, RMSE={mlp_val_metrics['RMSE']:.4f}")
    logger.info(f"XGBoost  -> R2={xgb_val_metrics['R2']:.4f}, MAE={xgb_val_metrics['MAE']:.4f}, MSE={xgb_val_metrics['MSE']:.4f}, RMSE={xgb_val_metrics['RMSE']:.4f}")
    
    metrics_order = ["R2", "MAE", "MSE", "RMSE"]
    best_weights = None
    best_val_r2 = -np.inf
    best_metric_name = None
    all_weight_schemes = {}
    
    for metric_name in metrics_order:
        weights = compute_weights(metric_name, mlp_val_metrics, xgb_val_metrics)
        ensemble_candidate = VotingRegressor([
            ('mlp', mlp_pipe),
            ('xgb', xgb_pipe)
        ], weights=list(weights))
        
        ensemble_candidate.fit(X_train, y_train)
        val_pred = ensemble_candidate.predict(X_val)
        val_metrics = evaluate_model(y_val, val_pred)
        
        # 记录所有权重方案
        all_weight_schemes[metric_name] = {
            'weights': weights,
            'validation_r2': val_metrics['R2'],
            'validation_rmse': val_metrics['RMSE'],
            'validation_mae': val_metrics['MAE']
        }
        
        logger.info(f"权重方案（{metric_name}） -> 权重 {weights}, 验证集 R2={val_metrics['R2']:.4f}")
        
        if val_metrics["R2"] > best_val_r2:
            best_val_r2 = val_metrics["R2"]
            best_weights = weights
            best_metric_name = metric_name
    
    if best_weights is None:
        best_weights = np.array([0.5, 0.5])
        best_metric_name = "Default"
    
    # 创建最佳集成模型
    best_ensemble = VotingRegressor([
        ('mlp', mlp_pipe),
        ('xgb', xgb_pipe)
    ], weights=list(best_weights))
    
    best_ensemble.fit(X_train, y_train)
    
    logger.info(f"最佳集成权重: {best_weights} (基于指标: {best_metric_name})")
    logger.info(f"集成模型验证集R2: {best_val_r2:.4f}")
    
    best_params = {
        'MLP': {
            'mlp__hidden_layer_sizes': (100, 100),
            'mlp__activation': 'relu',
            'mlp__alpha': 0.001,
            'mlp__learning_rate_init': 0.001,
            'mlp__batch_size': 64,
            'mlp__max_iter': 1000,
            'mlp__random_state': 42
        },
        'XGBoost': {
            'xgb__n_estimators': 200,
            'xgb__learning_rate': 0.15,
            'xgb__max_depth': 6,
            'xgb__subsample': 0.75,
            'xgb__colsample_bytree': 0.85,
            'xgb__gamma': 0.0,
            'xgb__reg_alpha': 0.0,
            'xgb__reg_lambda': 0.8,
            'xgb__objective': 'reg:squarederror',
            'xgb__random_state': 42
        },
        'Ensemble_Weight_Selection': {
            'selected_metric': best_metric_name,
            'final_weights': list(best_weights),
            'validation_r2': best_val_r2,
            'all_schemes': all_weight_schemes
        }
    }
    
    return best_ensemble, mlp_pipe, xgb_pipe, list(best_weights), best_params, best_metric_name, all_weight_schemes

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
    if model_name == "MLP":
        default_model_pipe = Pipeline([
            ('feature_selector', FeatureSelector(feature_cols)),
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(random_state=42))
        ])
    elif model_name == "XGBoost":
        default_model_pipe = Pipeline([
            ('feature_selector', FeatureSelector(feature_cols)),
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
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
    mlp_pipe, xgb_pipe = create_individual_models(feature_cols)
    
    # 训练个体模型
    mlp_pipe.fit(X_train, y_train_orig)
    xgb_pipe.fit(X_train, y_train_orig)
    
    # 获取个体模型的预测
    y_train_pred_mlp = mlp_pipe.predict(X_train)
    y_test_pred_mlp = mlp_pipe.predict(X_test)
    y_train_pred_xgb = xgb_pipe.predict(X_train)
    y_test_pred_xgb = xgb_pipe.predict(X_test)
    
    # 保存MLP模型的散点图数据
    save_scatter_data(y_train_orig, y_train_pred_mlp, y_test_orig, y_test_pred_mlp, 
                     X_train, X_test, y_train_orig, y_test_orig, feature_cols, "MLP")
    
    # 保存XGBoost模型的散点图数据
    save_scatter_data(y_train_orig, y_train_pred_xgb, y_test_orig, y_test_pred_xgb, 
                     X_train, X_test, y_train_orig, y_test_orig, feature_cols, "XGBoost")
    
    # 保存Ensemble模型的散点图数据
    save_scatter_data(y_train_orig, y_train_pred, y_test_orig, y_test_pred, 
                     X_train, X_test, y_train_orig, y_test_orig, feature_cols, "Ensemble")
    
    # 创建包含三个子图的图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # 定义清晰的颜色
    train_color = '#FF0000'  # 红色
    test_color = '#0000FF'   # 蓝色
    line_color = '#45B7D1'   # 青色
    
    # 绘制MLP模型的散点图
    ax1.scatter(y_train_orig, y_train_pred_mlp, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax1.scatter(y_test_orig, y_test_pred_mlp, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    all_y_mlp = np.concatenate([y_train_orig, y_test_orig])
    all_y_pred_mlp = np.concatenate([y_train_pred_mlp, y_test_pred_mlp])
    min_val_mlp = min(all_y_mlp.min(), all_y_pred_mlp.min())
    max_val_mlp = max(all_y_mlp.max(), all_y_pred_mlp.max())
    ax1.plot([min_val_mlp, max_val_mlp], [min_val_mlp, max_val_mlp], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax1.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax1.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax1.set_title('MLP Model', fontsize=8, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 设置坐标轴刻度字体加粗
    for tick in ax1.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    
    train_r2_mlp = r2_score(y_train_orig, y_train_pred_mlp)
    test_r2_mlp = r2_score(y_test_orig, y_test_pred_mlp)
    ax1.text(0.95, 0.05, f'Training R² = {train_r2_mlp:.3f}\nTest R² = {test_r2_mlp:.3f}', 
             transform=ax1.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    # 绘制XGBoost模型的散点图
    ax2.scatter(y_train_orig, y_train_pred_xgb, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax2.scatter(y_test_orig, y_test_pred_xgb, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    all_y_xgb = np.concatenate([y_train_orig, y_test_orig])
    all_y_pred_xgb = np.concatenate([y_train_pred_xgb, y_test_pred_xgb])
    min_val_xgb = min(all_y_xgb.min(), all_y_pred_xgb.min())
    max_val_xgb = max(all_y_xgb.max(), all_y_pred_xgb.max())
    ax2.plot([min_val_xgb, max_val_xgb], [min_val_xgb, max_val_xgb], 
             color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    ax2.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax2.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax2.set_title('XGBoost Model', fontsize=8, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    # 设置坐标轴刻度字体加粗
    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
    
    train_r2_xgb = r2_score(y_train_orig, y_train_pred_xgb)
    test_r2_xgb = r2_score(y_test_orig, y_test_pred_xgb)
    ax2.text(0.95, 0.05, f'Training R² = {train_r2_xgb:.3f}\nTest R² = {test_r2_xgb:.3f}', 
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
        'MLP': {
            'train': {
                'R2': train_r2_mlp,
                'MSE': mean_squared_error(y_train_orig, y_train_pred_mlp),
                'RMSE': np.sqrt(mean_squared_error(y_train_orig, y_train_pred_mlp)),
                'MAE': mean_absolute_error(y_train_orig, y_train_pred_mlp)
            },
            'test': {
                'R2': test_r2_mlp,
                'MSE': mean_squared_error(y_test_orig, y_test_pred_mlp),
                'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_test_pred_mlp)),
                'MAE': mean_absolute_error(y_test_orig, y_test_pred_mlp)
            }
        },
        'XGBoost': {
            'train': {
                'R2': train_r2_xgb,
                'MSE': mean_squared_error(y_train_orig, y_train_pred_xgb),
                'RMSE': np.sqrt(mean_squared_error(y_train_orig, y_train_pred_xgb)),
                'MAE': mean_absolute_error(y_train_orig, y_train_pred_xgb)
            },
            'test': {
                'R2': test_r2_xgb,
                'MSE': mean_squared_error(y_test_orig, y_test_pred_xgb),
                'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_test_pred_xgb)),
                'MAE': mean_absolute_error(y_test_orig, y_test_pred_xgb)
            }
        }
    }
    
    return individual_metrics

def save_results_to_excel(results, feature_cols, X_train, y_train, train_metrics, test_metrics, weight_selection_metric, all_weight_schemes):
    """保存结果到Excel文件 - 添加训练集指标和权重选择信息"""
    os.makedirs('ensemble_results', exist_ok=True)
    
    with pd.ExcelWriter('ensemble_results/ensemble_results.xlsx') as writer:
        # 1. 模型性能比较
        metrics_df = pd.DataFrame({
            'MLP': results['individual']['MLP']['test'],
            'XGBoost': results['individual']['XGBoost']['test'],
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
        
        # 5. 集成权重选择信息（新增）
        weight_selection_info = {
            'Selection_Method': ['基于验证集指标的自适应权重选择'],
            'Selected_Metric': [weight_selection_metric],
            'Final_Weights_MLP': [results['ensemble_weights'][0]],
            'Final_Weights_XGBoost': [results['ensemble_weights'][1]],
            'Validation_R2': [results['best_params']['Ensemble_Weight_Selection']['validation_r2']],
            'Weight_Calculation_Description': [
                f"基于{weight_selection_metric}指标计算权重：MLP权重={results['ensemble_weights'][0]:.4f}, XGBoost权重={results['ensemble_weights'][1]:.4f}"
            ]
        }
        weight_selection_df = pd.DataFrame(weight_selection_info)
        weight_selection_df.to_excel(writer, sheet_name='Ensemble Weight Selection', index=False)
        
        # 6. 所有权重方案比较（新增）
        schemes_data = []
        for metric_name, scheme_info in all_weight_schemes.items():
            schemes_data.append({
                'Metric': metric_name,
                'MLP_Weight': scheme_info['weights'][0],
                'XGBoost_Weight': scheme_info['weights'][1],
                'Validation_R2': scheme_info['validation_r2'],
                'Validation_RMSE': scheme_info['validation_rmse'],
                'Validation_MAE': scheme_info['validation_mae'],
                'Selected': 'Yes' if metric_name == weight_selection_metric else 'No'
            })
        
        schemes_comparison_df = pd.DataFrame(schemes_data)
        schemes_comparison_df.to_excel(writer, sheet_name='Weight Schemes Comparison', index=False)
        
        # 7. 特征信息
        feature_info = pd.DataFrame({
            'Feature': feature_cols,
            'Description': ['Molecular descriptor'] * len(feature_cols)
        })
        feature_info.to_excel(writer, sheet_name='Feature Information', index=False)
        
        # 8. 详细性能指标
        detailed_metrics = pd.DataFrame({
            'Model': ['MLP', 'XGBoost', 'Ensemble'],
            'Train_R2': [results['individual']['MLP']['train']['R2'], 
                        results['individual']['XGBoost']['train']['R2'], 
                        results['ensemble']['train']['R2']],
            'Test_R2': [results['individual']['MLP']['test']['R2'], 
                       results['individual']['XGBoost']['test']['R2'], 
                       results['ensemble']['test']['R2']],
            'Train_RMSE': [results['individual']['MLP']['train']['RMSE'], 
                          results['individual']['XGBoost']['train']['RMSE'], 
                          results['ensemble']['train']['RMSE']],
            'Test_RMSE': [results['individual']['MLP']['test']['RMSE'], 
                         results['individual']['XGBoost']['test']['RMSE'], 
                         results['ensemble']['test']['RMSE']],
            'Train_MAE': [results['individual']['MLP']['train']['MAE'], 
                         results['individual']['XGBoost']['train']['MAE'], 
                         results['ensemble']['train']['MAE']],
            'Test_MAE': [results['individual']['MLP']['test']['MAE'], 
                        results['individual']['XGBoost']['test']['MAE'], 
                        results['ensemble']['test']['MAE']]
        })
        detailed_metrics.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        
        # 9. 参数调优范围与默认值对比
        param_comparison_data = [
            # MLP参数
            {
                'Parameter': 'MLP__hidden_layer_sizes',
                'Tuning Range': '固定值',
                'Default Value': '(100, 100)',
                'Best Value': results['best_params']['MLP'].get('mlp__hidden_layer_sizes', 'N/A')
            },
            {
                'Parameter': 'MLP__activation',
                'Tuning Range': '固定值',
                'Default Value': 'relu',
                'Best Value': results['best_params']['MLP'].get('mlp__activation', 'N/A')
            },
            {
                'Parameter': 'MLP__alpha',
                'Tuning Range': '固定值',
                'Default Value': '0.001',
                'Best Value': results['best_params']['MLP'].get('mlp__alpha', 'N/A')
            },
            {
                'Parameter': 'MLP__learning_rate_init',
                'Tuning Range': '固定值',
                'Default Value': '0.001',
                'Best Value': results['best_params']['MLP'].get('mlp__learning_rate_init', 'N/A')
            },
            {
                'Parameter': 'MLP__batch_size',
                'Tuning Range': '固定值',
                'Default Value': '64',
                'Best Value': results['best_params']['MLP'].get('mlp__batch_size', 'N/A')
            },
            {
                'Parameter': 'MLP__max_iter',
                'Tuning Range': '固定值',
                'Default Value': '1000',
                'Best Value': results['best_params']['MLP'].get('mlp__max_iter', 'N/A')
            },
            # XGBoost参数
            {
                'Parameter': 'XGBoost__n_estimators',
                'Tuning Range': '固定值',
                'Default Value': '200',
                'Best Value': results['best_params']['XGBoost'].get('xgb__n_estimators', 'N/A')
            },
            {
                'Parameter': 'XGBoost__learning_rate',
                'Tuning Range': '固定值',
                'Default Value': '0.15',
                'Best Value': results['best_params']['XGBoost'].get('xgb__learning_rate', 'N/A')
            },
            {
                'Parameter': 'XGBoost__max_depth',
                'Tuning Range': '固定值',
                'Default Value': '6',
                'Best Value': results['best_params']['XGBoost'].get('xgb__max_depth', 'N/A')
            },
            {
                'Parameter': 'XGBoost__subsample',
                'Tuning Range': '固定值',
                'Default Value': '0.75',
                'Best Value': results['best_params']['XGBoost'].get('xgb__subsample', 'N/A')
            },
            {
                'Parameter': 'XGBoost__colsample_bytree',
                'Tuning Range': '固定值',
                'Default Value': '0.85',
                'Best Value': results['best_params']['XGBoost'].get('xgb__colsample_bytree', 'N/A')
            },
            {
                'Parameter': 'XGBoost__gamma',
                'Tuning Range': '固定值',
                'Default Value': '0.0',
                'Best Value': results['best_params']['XGBoost'].get('xgb__gamma', 'N/A')
            },
            {
                'Parameter': 'XGBoost__reg_alpha',
                'Tuning Range': '固定值',
                'Default Value': '0.0',
                'Best Value': results['best_params']['XGBoost'].get('xgb__reg_alpha', 'N/A')
            },
            {
                'Parameter': 'XGBoost__reg_lambda',
                'Tuning Range': '固定值',
                'Default Value': '0.8',
                'Best Value': results['best_params']['XGBoost'].get('xgb__reg_lambda', 'N/A')
            },
            # 集成权重参数
            {
                'Parameter': 'Ensemble Weight Selection',
                'Tuning Range': '基于验证指标自适应',
                'Default Value': '基于R2/MAE/MSE/RMSE计算',
                'Best Value': f"基于{weight_selection_metric}指标，权重=[{results['ensemble_weights'][0]:.4f}, {results['ensemble_weights'][1]:.4f}]"
            }
        ]
        
        param_comparison_df = pd.DataFrame(param_comparison_data)
        param_comparison_df.to_excel(writer, sheet_name='Parameter Comparison', index=False)
        
        # 10. 参数调优统计
        tuning_stats = pd.DataFrame({
            'Metric': ['Total Parameter Configurations', 'Weight Schemes Evaluated', 'Selected Weight Scheme'],
            'Value': ['Fixed best parameters', '4', weight_selection_metric],
            'Description': [
                'MLP与XGBoost均使用固定最佳参数', 
                '基于不同指标计算的权重方案数量', 
                f'最终选择的权重方案基于{weight_selection_metric}指标'
            ]
        })
        tuning_stats.to_excel(writer, sheet_name='Tuning Statistics', index=False)


def main():
    try:
        input_file = "CaCo-2_molecular_descriptors_reduced.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info(f"开始处理文件: {input_file}")
        start_time = time.time()
        
        # 加载和准备数据
        X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(input_file)
        
        # 优化集成模型 - 现在返回更多信息
        ensemble_model, mlp_model, xgb_model, best_weights, best_params, best_metric_name, all_weight_schemes = optimize_ensemble_model(
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
            'best_params': best_params,
            'ensemble_weights': best_weights
        }
        
        # 保存结果到Excel - 传入权重选择信息
        save_results_to_excel(results, feature_cols, X_train, y_train, train_metrics, test_metrics, 
                             best_metric_name, all_weight_schemes)
        
        # 保存模型
        os.makedirs('saved_models', exist_ok=True)
        joblib.dump(ensemble_model, 'saved_models/ensemble_model.pkl')
        
        # 保存个体模型
        joblib.dump(mlp_model, 'saved_models/mlp_best_model.pkl')
        joblib.dump(xgb_model, 'saved_models/xgb_best_model.pkl')
        
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
        logger.info(f"最佳集成权重: {best_weights} (基于指标: {best_metric_name})")
        logger.info(f"集成模型测试集R2: {test_metrics['R2']:.4f}")
        logger.info(f"集成模型测试集RMSE: {test_metrics['RMSE']:.4f}")
        logger.info(f"MLP测试集R2: {individual_metrics['MLP']['test']['R2']:.4f}")
        logger.info(f"XGBoost测试集R2: {individual_metrics['XGBoost']['test']['R2']:.4f}")
        logger.info("="*50)
        
        # 打印参数调优信息
        logger.info("\n参数调优信息:")
        logger.info("MLP参数：固定使用训练得到的最佳配置")
        logger.info("XGBoost参数：固定使用训练得到的最佳配置")
        logger.info(f"集成权重调优: 基于验证指标自适应，最终选择{best_metric_name}指标")
        
        # 执行时间
        elapsed = time.time() - start_time
        logger.info(f"\n总执行时间: {elapsed:.2f}秒")
        logger.info("处理完成!")
        
        return results, ensemble_model, feature_importance_df, shap_values, best_metric_name, all_weight_schemes
    
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise

if __name__ == "__main__":
    results, ensemble_model, feature_importance_df, shap_values, best_metric_name, all_weight_schemes = main()