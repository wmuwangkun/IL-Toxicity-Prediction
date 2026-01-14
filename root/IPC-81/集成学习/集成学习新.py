
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    'font.size': 6,
    'axes.titlesize': 7,
    'axes.labelsize': 6,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.titlesize': 8,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
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
        df = pd.read_excel(file_path)
        logger.info(f"数据加载成功: {df.shape[0]}行, {df.shape[1]}列")
        
        target_col = 'log_EC50'
        possible_target_names = ['log_EC50', 'log EC50', 'logEC50', 'EC50_log', 'log10(EC50)']
        for name in possible_target_names:
            if name in df.columns:
                target_col = name
                break
        
        smiles_col = None
        possible_smiles_names = ['Canonical SMILES', 'SMILES', 'smiles', 'canonical_smiles']
        for name in possible_smiles_names:
            if name in df.columns:
                smiles_col = name
                break
        
        non_feature_cols = ['Name', 'Empirical formula', 'Canonical SMILES', 'SMILES', 'smiles', 
                           'canonical_smiles', 'log_EC50', 'log EC50', 'logEC50', 'EC50_log', 'log10(EC50)']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        if not feature_cols:
            raise ValueError("未找到特征列！请检查数据格式")
        
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"目标变量: {target_col}")
        logger.info(f"特征数量: {len(feature_cols)}")
        logger.info(f"特征示例: {feature_cols[:5]}...")
        
        if smiles_col:
            logger.info(f"找到SMILES列: {smiles_col}")
        
        X = X.fillna(X.median())
        
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
    svr_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=17.58364027, gamma=0.00190696, epsilon=0.1))
    ])
    
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
    svr_pipe, enet_pipe = create_individual_models(feature_cols)
    logger.info("使用预设最佳参数训练SVR模型...")
    svr_pipe.fit(X_train, y_train)
    
    logger.info("使用预设最佳参数训练ElasticNet模型...")
    enet_pipe.fit(X_train, y_train)
    
    best_svr = svr_pipe
    best_enet = enet_pipe
    
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
    best_metric_name = None
    all_weight_schemes = {}  # 记录所有权重方案
    
    for metric_name in metrics_order:
        weights = compute_weights(metric_name, svr_val_metrics, enet_val_metrics)
        ensemble_candidate = VotingRegressor([
            ('svr', best_svr),
            ('enet', best_enet)
        ], weights=list(weights))
        
        ensemble_candidate.fit(X_train, y_train)
        val_pred = ensemble_candidate.predict(X_val)
        val_metrics = evaluate_model(y_val, val_pred)
        
        # 记录每个权重方案的信息
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
    
    best_ensemble = VotingRegressor([
        ('svr', best_svr),
        ('enet', best_enet)
    ], weights=list(best_weights))
    
    best_ensemble.fit(X_train, y_train)
    
    best_params = {
        'SVR': {
            'svr__kernel': 'rbf',
            'svr__C': 17.58364027,
            'svr__gamma': 0.00190696,
            'svr__epsilon': 0.1
        },
        'ElasticNet': {
            'enet__alpha': 0.0376,
            'enet__l1_ratio': 0.376,
            'enet__max_iter': 10000,
            'enet__tol': 0.0001
        },
        'Ensemble_Weight_Selection': {
            'selected_metric': best_metric_name,
            'final_weights': list(best_weights),
            'validation_r2': best_val_r2,
            'all_schemes': all_weight_schemes
        }
    }
    
    logger.info(f"SVR最佳参数: {best_params['SVR']}")
    logger.info(f"弹性网络最佳参数: {best_params['ElasticNet']}")
    logger.info(f"最佳集成权重: {best_weights} (基于指标: {best_metric_name})")
    logger.info(f"集成模型验证集R2: {best_val_r2:.4f}")
    
    return best_ensemble, best_svr, best_enet, list(best_weights), best_params, best_metric_name, all_weight_schemes

def evaluate_model(y_true, y_pred):
    """评估模型性能 - 添加MRE指标"""
    mre = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
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
    
    explainer = shap.KernelExplainer(ensemble_model.predict, X_train.values[:100])
    shap_values = explainer.shap_values(X_test.values[:50])
    
    mean_shap_abs = np.mean(np.abs(shap_values), axis=0)
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_ABS_SHAP': mean_shap_abs
    }).sort_values('Mean_ABS_SHAP', ascending=True)
    
    top_features = feature_importance_df.tail(15)
    
    fig_width = 6
    fig_height = 8
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    bars = ax.barh(range(len(top_features)), 
                   top_features['Mean_ABS_SHAP'],
                   color=colors, alpha=0.85, edgecolor='none', linewidth=0)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=12, fontweight='bold')
    
    ax.set_xlabel('mean(|SHAP value|)', fontsize=12, fontweight='bold')
    
    ax.tick_params(axis='x', labelsize=12, width=1.2, length=4, labelcolor='#2E2E2E')
    ax.tick_params(axis='y', width=1.2, length=4)
    
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        label.set_color('#000000')
    
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        label.set_color('#000000')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    for i, (bar, value) in enumerate(zip(bars, top_features['Mean_ABS_SHAP'])):
        ax.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=12, 
                fontweight='bold', color='#2E2E2E')
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.1)
    
    plt.savefig(f'ensemble_results/{model_name}_global_explanation.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'ensemble_results/{model_name}_global_explanation.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'ensemble_results/{model_name}_global_explanation.tif', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='tiff')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    shap.summary_plot(shap_values, X_test.values[:50],
                      feature_names=feature_cols,
                      plot_type="dot",
                      show=False,
                      max_display=15)
    
    ax = plt.gca()
    
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    
    ax.tick_params(axis='x', labelsize=12, width=1.2, length=4, labelcolor='#2E2E2E')
    ax.tick_params(axis='y', labelsize=12, width=1.2, length=4)
    
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        label.set_color('#000000')
    
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
        label.set_color('#000000')
    
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=12)
    for label in cbar.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    cbar.set_ylabel('')
    
    plt.draw()
    
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        label.set_color('#000000')
    
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
        label.set_color('#000000')
    
    ax.set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    ax.grid(False)
    
    plt.subplots_adjust(left=0.25, right=0.85, top=0.95, bottom=0.1)
    
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
    
    if model_name == "SVR":
        default_model_pipe = Pipeline([
            ('feature_selector', FeatureSelector(feature_cols)),
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
    elif model_name == "ElasticNet":
        default_model_pipe = Pipeline([
            ('feature_selector', FeatureSelector(feature_cols)),
            ('scaler', StandardScaler()),
            ('enet', ElasticNet())
        ])
    else:  # Ensemble
        default_model_pipe = None
    
    if default_model_pipe is not None:
        default_model_pipe.fit(X_train, y_train_orig)
        y_train_pred_default = default_model_pipe.predict(X_train)
        y_test_pred_default = default_model_pipe.predict(X_test)
    else:
        y_train_pred_default = np.nan * np.ones_like(y_train_pred)
        y_test_pred_default = np.nan * np.ones_like(y_test_pred)
    
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
    
    with pd.ExcelWriter(f'ensemble_results/{model_name}_scatter_plot_data.xlsx') as writer:
        full_data.to_excel(writer, sheet_name='All_Data', index=False)
        scatter_data.to_excel(writer, sheet_name='Training_Set', index=False)
        test_data.to_excel(writer, sheet_name='Test_Set', index=False)
        
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
    """绘制集成模型适用域分析图并保存数据"""
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

    ax.set_xlabel('Average Feature Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standardized Residuals', fontsize=12, fontweight='bold')
    ax.set_title('Applicability Domain Analysis (Training & Testing)', fontsize=12, fontweight='bold')

    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')

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
        prop={'weight': 'bold', 'size': 10}
    )
    ax.grid(True, linestyle='--', alpha=0.3)

    exceed_text = (
        f"Training outside: {((ad_df['Set'] == 'Training') & ad_df['OutOfDomain']).sum()}\n"
        f"Testing outside: {((ad_df['Set'] == 'Testing') & ad_df['OutOfDomain']).sum()}"
    )
    ax.text(
        0.013,
        0.983,
        exceed_text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        fontfamily='Arial',
        color='darkred',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=0.8)
    )

    plt.tight_layout()
    plt.savefig('ensemble_results/ensemble_applicability_domain.png', dpi=300, bbox_inches='tight')
    plt.savefig('ensemble_results/ensemble_applicability_domain.pdf', bbox_inches='tight')
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
    ax3.set_title('Ensemble', fontsize=8, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
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
    
    plt.savefig(f'ensemble_results/{model_name}_comparison_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'ensemble_results/{model_name}_comparison_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"对比散点图已保存至 ensemble_results/{model_name}_comparison_scatter_plot.png")
    
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
    
    plt.savefig(f'ensemble_results/Ensemble_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'ensemble_results/Ensemble_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"单独的集成模型散点图已保存至 ensemble_results/Ensemble_scatter_plot.png")
    
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

def save_results_to_excel(results, feature_cols, X_train, y_train, train_metrics, test_metrics, weight_selection_metric, all_weight_schemes):
    """保存结果到Excel文件 - 添加训练集指标和权重选择信息"""
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
        
        # 5. 集成权重选择信息（新增）
        weight_selection_info = {
            'Selection_Method': ['基于验证集指标的自适应权重选择'],
            'Selected_Metric': [weight_selection_metric],
            'Final_Weights_SVR': [results['ensemble_weights'][0]],
            'Final_Weights_ElasticNet': [results['ensemble_weights'][1]],
            'Validation_R2': [results['best_params']['Ensemble_Weight_Selection']['validation_r2']],
            'Weight_Calculation_Description': [
                f"基于{weight_selection_metric}指标计算权重：SVR权重={results['ensemble_weights'][0]:.4f}, ElasticNet权重={results['ensemble_weights'][1]:.4f}"
            ]
        }
        weight_selection_df = pd.DataFrame(weight_selection_info)
        weight_selection_df.to_excel(writer, sheet_name='Ensemble Weight Selection', index=False)
        
        # 6. 所有权重方案比较（新增）
        schemes_data = []
        for metric_name, scheme_info in all_weight_schemes.items():
            schemes_data.append({
                'Metric': metric_name,
                'SVR_Weight': scheme_info['weights'][0],
                'ElasticNet_Weight': scheme_info['weights'][1],
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
                         results['ensemble']['test']['RMSE']],
            'Train_MAE': [results['individual']['SVR']['train']['MAE'], 
                         results['individual']['ElasticNet']['train']['MAE'], 
                         results['ensemble']['train']['MAE']],
            'Test_MAE': [results['individual']['SVR']['test']['MAE'], 
                        results['individual']['ElasticNet']['test']['MAE'], 
                        results['ensemble']['test']['MAE']]
        })
        detailed_metrics.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        
        # 9. 参数调优范围与默认值对比
        param_comparison_data = [
            {
                'Parameter': 'SVR__C',
                'Tuning Range': '固定最佳值',
                'Default Value': '17.58364027',
                'Best Value': results['best_params']['SVR'].get('svr__C', 'N/A')
            },
            {
                'Parameter': 'SVR__gamma',
                'Tuning Range': '固定最佳值',
                'Default Value': '0.00190696',
                'Best Value': results['best_params']['SVR'].get('svr__gamma', 'N/A')
            },
            {
                'Parameter': 'SVR__epsilon',
                'Tuning Range': '固定最佳值',
                'Default Value': '0.1',
                'Best Value': results['best_params']['SVR'].get('svr__epsilon', 'N/A')
            },
            {
                'Parameter': 'ElasticNet__alpha',
                'Tuning Range': '固定最佳值',
                'Default Value': '0.0376',
                'Best Value': results['best_params']['ElasticNet'].get('enet__alpha', 'N/A')
            },
            {
                'Parameter': 'ElasticNet__l1_ratio',
                'Tuning Range': '固定最佳值',
                'Default Value': '0.376',
                'Best Value': results['best_params']['ElasticNet'].get('enet__l1_ratio', 'N/A')
            },
            {
                'Parameter': 'ElasticNet__max_iter',
                'Tuning Range': '固定最佳值',
                'Default Value': '10000',
                'Best Value': results['best_params']['ElasticNet'].get('enet__max_iter', 'N/A')
            },
            {
                'Parameter': 'ElasticNet__tol',
                'Tuning Range': '固定最佳值',
                'Default Value': '0.0001',
                'Best Value': results['best_params']['ElasticNet'].get('enet__tol', 'N/A')
            },
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
                'SVR与ElasticNet均使用固定最佳参数', 
                '基于不同指标计算的权重方案数量', 
                f'最终选择的权重方案基于{weight_selection_metric}指标'
            ]
        })
        tuning_stats.to_excel(writer, sheet_name='Tuning Statistics', index=False)

def main():
    try:
        input_file = "IPC-81_molecular_descriptors_reduced.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info(f"开始处理文件: {input_file}")
        start_time = time.time()
        
        X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(input_file)
        
        # 优化集成模型 - 现在返回更多信息
        ensemble_model, svr_model, enet_model, best_weights, best_params, best_metric_name, all_weight_schemes = optimize_ensemble_model(
            X_train, y_train, X_test, y_test, feature_cols
        )
        
        y_train_pred = ensemble_model.predict(X_train)
        train_metrics = evaluate_model(y_train, y_train_pred)
        
        y_test_pred = ensemble_model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_test_pred)
        
        ensemble_test_residuals = (y_test - y_test_pred).to_numpy()
        plot_residual_histogram(ensemble_test_residuals, 'ensemble_tuned_test', 'ensemble_results')
        
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
        
        individual_metrics = plot_scatter(y_train, y_train_pred, y_test, y_test_pred, "Ensemble", 
                                        X_train, X_test, y_train, y_test, feature_cols)
        
        plot_applicability_domain(ensemble_model, X_train, X_test, y_train, y_test, feature_cols)
        
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
        
        os.makedirs('saved_models', exist_ok=True)
        joblib.dump(ensemble_model, 'saved_models/ensemble_model.pkl')
        joblib.dump(svr_model, 'saved_models/svr_best_model.pkl')
        joblib.dump(enet_model, 'saved_models/enet_best_model.pkl')
        
        logger.info("集成模型已保存至 saved_models/ 目录")
        logger.info("个体模型已保存至 saved_models/ 目录")
        
        logger.info("\n=== 开始SHAP分析 ===")
        feature_importance_df, shap_values = create_shap_analysis(
            X_train, X_test, y_train, y_test, feature_cols, ensemble_model, "Ensemble"
        )
        
        logger.info("\n=== 前15个最重要特征 ===")
        top_15_features = feature_importance_df.head(15)
        for i, (idx, row) in enumerate(top_15_features.iterrows(), 1):
            logger.info(f"{i:2d}. {row['Feature']:<25} SHAP值: {row['Mean_ABS_SHAP']:.4f}")
        
        logger.info("\n" + "="*50)
        logger.info("最终结果摘要")
        logger.info("="*50)
        logger.info(f"最佳集成权重: {best_weights} (基于指标: {best_metric_name})")
        logger.info(f"集成模型测试集R2: {test_metrics['R2']:.4f}")
        logger.info(f"集成模型测试集RMSE: {test_metrics['RMSE']:.4f}")
        logger.info(f"SVR测试集R2: {individual_metrics['SVR']['test']['R2']:.4f}")
        logger.info(f"ElasticNet测试集R2: {individual_metrics['ElasticNet']['test']['R2']:.4f}")
        logger.info("="*50)
        
        logger.info("\n模型参数信息:")
        logger.info("SVR使用参数:")
        logger.info("  kernel: rbf")
        logger.info("  C: 17.58364027")
        logger.info("  gamma: 0.00190696")
        logger.info("  epsilon: 0.1")
        logger.info("ElasticNet使用参数:")
        logger.info("  alpha: 0.0376")
        logger.info("  l1_ratio: 0.376")
        logger.info("  max_iter: 10000")
        logger.info("  tol: 0.0001")
        logger.info(f"集成权重调优: 基于验证指标自适应，最终选择{best_metric_name}指标")
        
        elapsed = time.time() - start_time
        logger.info(f"\n总执行时间: {elapsed:.2f}秒")
        logger.info("处理完成!")
        
        return results, ensemble_model, feature_importance_df, shap_values, best_metric_name, all_weight_schemes
    
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise

if __name__ == "__main__":
    results, ensemble_model, feature_importance_df, shap_values, best_metric_name, all_weight_schemes = main()