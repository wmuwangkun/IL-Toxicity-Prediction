# train_random_forest.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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
from matplotlib.font_manager import FontProperties

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
        logging.FileHandler("random_forest_tuning.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def train_and_tune_rf(X_train, X_test, y_train, y_test):
    """训练和调优随机森林模型"""
    # 默认模型
    default_rf = RandomForestRegressor(random_state=42)
    default_rf.fit(X_train, y_train)
    
    # 训练集和测试集预测
    y_train_pred_default = default_rf.predict(X_train)
    y_test_pred_default = default_rf.predict(X_test)
    
    # 评估默认模型
    default_train_metrics = evaluate_model(y_train, y_train_pred_default)
    default_test_metrics = evaluate_model(y_test, y_test_pred_default)
    
    # 记录默认参数
    default_params = default_rf.get_params()
    
    # 参数网格
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    logger.info("开始网格搜索...")
    start_time = time.time()
    
    # 网格搜索
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"网格搜索完成，耗时: {time.time() - start_time:.2f}秒")
    logger.info(f"最佳参数: {grid_search.best_params_}")
    logger.info(f"最佳交叉验证R²: {grid_search.best_score_:.4f}")
    
    # 最佳模型
    best_rf = grid_search.best_estimator_
    y_train_pred_best = best_rf.predict(X_train)
    y_test_pred_best = best_rf.predict(X_test)
    
    # 评估最佳模型
    best_train_metrics = evaluate_model(y_train, y_train_pred_best)
    best_test_metrics = evaluate_model(y_test, y_test_pred_best)
    
    # 记录最佳参数
    best_params = best_rf.get_params()
    
    # 保存结果 - 包含训练集和测试集指标
    results = {
        'default_train': default_train_metrics,
        'default_test': default_test_metrics,
        'best_train': best_train_metrics,
        'best_test': best_test_metrics,
        'best_params': grid_search.best_params_,
        'param_grid': param_grid,
        'cv_results': grid_search.cv_results_,
        'default_params': default_params,
        'best_full_params': best_params
    }
    
    return results, default_rf, best_rf

def plot_results(results, X_train, X_test, y_train, y_test):
    """绘制结果图表 - SCI风格"""
    os.makedirs('rf_results', exist_ok=True)
    
    # 重新训练默认模型以获取训练集预测
    default_rf = RandomForestRegressor(random_state=42)
    default_rf.fit(X_train, y_train)
    y_train_pred_default = default_rf.predict(X_train)
    y_test_pred_default = default_rf.predict(X_test)
    
    # 重新训练最佳模型以获取训练集预测
    best_rf = RandomForestRegressor(** results['best_params'], random_state=42)
    best_rf.fit(X_train, y_train)
    y_train_pred_best = best_rf.predict(X_train)
    y_test_pred_best = best_rf.predict(X_test)
    
    # 定义颜色
    train_color = '#FF0000'  # 红色
    test_color = '#0000FF'   # 蓝色
    line_color = '#45B7D1'   # 青色
    
    # 1. 创建对比图（两个子图）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # 默认模型散点图
    ax1.scatter(y_train, y_train_pred_default, alpha=0.8, s=20, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax1.scatter(y_test, y_test_pred_default, alpha=0.8, s=20, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    
    # 理想预测线
    min_val = min(min(y_train), min(y_test), min(y_train_pred_default), min(y_test_pred_default))
    max_val = max(max(y_train), max(y_test), max(y_train_pred_default), max(y_test_pred_default))
    ax1.plot([min_val, max_val], [min_val, max_val], color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    # 修改坐标轴标签 - 加粗字体
    ax1.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax1.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax1.set_title('Default Parameters - Random Forest', fontsize=8, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴刻度标签加粗
    for tick in ax1.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    
    ax1.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
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
    
    ax2.plot([min_val, max_val], [min_val, max_val], color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    # 修改坐标轴标签 - 加粗字体
    ax2.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax2.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax2.set_title('Tuned Parameters - Random Forest', fontsize=8, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴刻度标签加粗
    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
    
    ax2.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    
    train_r2_best = r2_score(y_train, y_train_pred_best)
    test_r2_best = r2_score(y_test, y_test_pred_best)
    ax2.text(0.95, 0.05, f'Training R² = {train_r2_best:.3f}\nTest R² = {test_r2_best:.3f}', 
             transform=ax2.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    
    plt.tight_layout()
    plt.savefig('rf_results/rf_comparison_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('rf_results/rf_comparison_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. 单独保存默认模型图
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_train, y_train_pred_default, alpha=0.8, s=25, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test, y_test_pred_default, alpha=0.8, s=25, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    ax.plot([min_val, max_val], [min_val, max_val], color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    # 修改坐标轴标签 - 加粗字体
    ax.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax.set_title('Default Random Forest', fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴刻度标签加粗
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    ax.text(0.95, 0.05, f'Training R² = {train_r2_default:.3f}\nTest R² = {test_r2_default:.3f}', 
             transform=ax.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    plt.savefig('rf_results/rf_default_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('rf_results/rf_default_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. 单独保存调优模型图 - 修改标题和坐标轴标签
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(y_train, y_train_pred_best, alpha=0.8, s=25, color=train_color, 
               edgecolors='white', linewidth=0.5, label='Training set')
    ax.scatter(y_test, y_test_pred_best, alpha=0.8, s=25, color=test_color, 
               edgecolors='white', linewidth=0.5, label='Test set')
    ax.plot([min_val, max_val], [min_val, max_val], color=line_color, linestyle='--', lw=1.5, alpha=0.9)
    
    # 修改坐标轴标签 - 加粗字体
    ax.set_xlabel('Experimental values', fontsize=7, fontweight='bold')
    ax.set_ylabel('Predicted values', fontsize=7, fontweight='bold')
    ax.set_title('RF', fontsize=8, fontweight='bold')  # 修改标题为RF
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 坐标轴刻度标签加粗
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    ax.text(0.95, 0.05, f'Training R² = {train_r2_best:.3f}\nTest R² = {test_r2_best:.3f}', 
             transform=ax.transAxes, fontsize=6, verticalalignment='bottom', 
             horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=0.5))
    plt.savefig('rf_results/rf_tuned_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('rf_results/rf_tuned_scatter_plot.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("散点图已保存至 rf_results/ 目录")

def _calculate_average_feature_distance(train_features, other_features=None):
    """计算平均特征距离"""
    if other_features is None:
        distance_matrix = pairwise_distances(train_features, metric='euclidean')
        np.fill_diagonal(distance_matrix, np.nan)
        return np.nanmean(distance_matrix, axis=1)
    distance_matrix = pairwise_distances(other_features, train_features, metric='euclidean')
    return distance_matrix.mean(axis=1)

def plot_applicability_domain(results, X_train, X_test, y_train, y_test):
    """绘制适用域分析图 - 调整为与参考代码一致的风格"""
    os.makedirs('rf_results', exist_ok=True)

    best_rf = RandomForestRegressor(**results['best_params'], random_state=42)
    best_rf.fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    avg_dist_train = _calculate_average_feature_distance(X_train_scaled)
    avg_dist_test = _calculate_average_feature_distance(X_train_scaled, X_test_scaled)

    y_train_pred = best_rf.predict(X_train)
    y_test_pred = best_rf.predict(X_test)

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
    plt.savefig('rf_results/rf_applicability_domain.png', dpi=300, bbox_inches='tight')
    plt.savefig('rf_results/rf_applicability_domain.tif', dpi=300, bbox_inches='tight', format='tiff')
    plt.close()

    testing_outliers = ((ad_df['Set'] == 'Testing') & ad_df['OutOfDomain']).sum()
    summary_df = pd.DataFrame({
        'Metric': ['Testing Samples Outside Domain'],
        'Count': [testing_outliers]
    })

    with pd.ExcelWriter('rf_results/rf_applicability_domain_data.xlsx') as writer:
        ad_df.to_excel(writer, sheet_name='All_Samples', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    logger.info("适用域分析图已保存至 rf_results/ 目录")

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

def save_scatter_data(results, X_train, X_test, y_train, y_test):
    """保存散点图数据到Excel文件"""
    os.makedirs('rf_results', exist_ok=True)
    
    # 重新训练默认模型以获取预测值
    default_rf = RandomForestRegressor(random_state=42)
    default_rf.fit(X_train, y_train)
    y_train_pred_default = default_rf.predict(X_train)
    y_test_pred_default = default_rf.predict(X_test)
    
    # 重新训练最佳模型以获取预测值
    best_rf = RandomForestRegressor(** results['best_params'], random_state=42)
    best_rf.fit(X_train, y_train)
    y_train_pred_best = best_rf.predict(X_train)
    y_test_pred_best = best_rf.predict(X_test)
    
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
    with pd.ExcelWriter('rf_results/scatter_plot_data.xlsx') as writer:
        full_data.to_excel(writer, sheet_name='All_Data', index=False)
        
        # 分别保存训练集和测试集
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
    
    logger.info("散点图数据已保存至 rf_results/scatter_plot_data.xlsx")

def save_results_to_excel(results, feature_cols, X_train, y_train):
    """保存结果到Excel文件 - 增加训练集指标和MRE"""
    os.makedirs('rf_results', exist_ok=True)
    
    # 创建Excel写入器
    with pd.ExcelWriter('rf_results/random_forest_results.xlsx') as writer:
        # 1. 模型性能比较 - 包含训练集和测试集
        # 创建性能比较表
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
        
        # 8. 特征重要性
        best_rf = RandomForestRegressor(**results['best_params'], random_state=42)
        best_rf.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
        
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
        
        # 10. 网格搜索详细结果
        cv_results = results['cv_results']
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_excel(writer, sheet_name='CV Results', index=False)
        
        # 11. 训练集vs测试集性能对比
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
        input_file = "IPC-81_molecular_descriptors_reduced.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info(f"开始处理文件: {input_file}")
        start_time = time.time()
        
        # 加载和准备数据
        X_train, X_test, y_train, y_test, feature_cols, df = load_and_prepare_data(input_file)
        
        # 训练和调优模型
        results, default_rf, best_rf = train_and_tune_rf(X_train, X_test, y_train, y_test)

        # 绘制残差直方图（仅调优模型测试集）
        tuned_test_residuals = (y_test - best_rf.predict(X_test)).to_numpy()
        plot_residual_histogram(tuned_test_residuals, 'rf_tuned_test', 'rf_results')
        
        # 绘制结果
        plot_results(results, X_train, X_test, y_train, y_test)
        plot_applicability_domain(results, X_train, X_test, y_train, y_test)
        
        # 保存散点图数据到Excel
        save_scatter_data(results, X_train, X_test, y_train, y_test)
        
        # 保存结果到Excel
        save_results_to_excel(results, feature_cols, X_train, y_train)
        
        # 保存模型
        joblib.dump(default_rf, 'rf_results/default_rf_model.pkl')
        joblib.dump(best_rf, 'rf_results/tuned_rf_model.pkl')
        
        # 记录执行时间
        elapsed = time.time() - start_time
        logger.info(f"\n总执行时间: {elapsed:.2f}秒")
        logger.info("随机森林模型调优完成!")
        
        # 打印最终结果摘要
        print("\n" + "="*80)
        print("RANDOM FOREST TUNING RESULTS SUMMARY")
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