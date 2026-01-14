import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ablation_study.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_mre(y_true, y_pred):
    """计算平均相对误差 (Mean Relative Error)"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def analyze_descriptor_types():
    """分析描述符类型并输出到Excel"""
    # 根据您提供的数据列分析
    columns = [
        "Name", "Empirical formula", "Canonical SMILES", "log_EC50", 
        "MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "HeavyAtomCount", "RingCount", 
        "BalabanJ", "BertzCT", "Chi0n", "Chi1n", "Chi2n", "Kappa1", "Kappa2", "HallKierAlpha", 
        "FractionCSP3", "NumRadicalElectrons", "NumValenceElectrons", "NumAmideBonds", 
        "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAliphaticCarbocycles", 
        "NumAliphaticHeterocycles", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", 
        "LipinskiHBA", "LipinskiHBD", "LipinskiRuleOf5Failures", "SLogP", "LabuteASA", 
        "PEOE_VSA1", "fr_Al_OH", "fr_Ar_OH", "fr_COO", "fr_NH0", "fr_NH1", "fr_NH2", "fr_SH", "fr_halogen", 
        "MACCS_PC1", "MACCS_PC2", "MACCS_PC3", "MACCS_PC4", "MACCS_PC5", "MACCS_PC6", "MACCS_PC7", 
        "MACCS_PC8", "MACCS_PC9", "MACCS_PC10", "MACCS_PC11", "MACCS_PC12", "MACCS_PC13", "MACCS_PC14", 
        "MACCS_PC15", "MACCS_PC16", "MACCS_PC17", "MACCS_PC18", "MACCS_PC19", "MACCS_PC20", "MACCS_PC21", 
        "MACCS_PC22", "MACCS_PC23", "MACCS_PC24", "MACCS_PC25", "MACCS_PC26", "MACCS_PC27", "MACCS_PC28", 
        "MACCS_PC29", "MACCS_PC30", "MACCS_PC31", "MACCS_PC32", 
        "ECFP_PC1", "ECFP_PC2", "ECFP_PC3", "ECFP_PC4", "ECFP_PC5", "ECFP_PC6", "ECFP_PC7", "ECFP_PC8", 
        "ECFP_PC9", "ECFP_PC10", "ECFP_PC11", "ECFP_PC12", "ECFP_PC13", "ECFP_PC14", "ECFP_PC15", "ECFP_PC16", 
        "ECFP_PC17", "ECFP_PC18", "ECFP_PC19", "ECFP_PC20", "ECFP_PC21", "ECFP_PC22", "ECFP_PC23", "ECFP_PC24", 
        "ECFP_PC25", "ECFP_PC26", "ECFP_PC27", "ECFP_PC28", "ECFP_PC29", "ECFP_PC30", "ECFP_PC31", "ECFP_PC32", 
        "ECFP_PC33", "ECFP_PC34", "ECFP_PC35", "ECFP_PC36", "ECFP_PC37", "ECFP_PC38", "ECFP_PC39", "ECFP_PC40", 
        "ECFP_PC41", "ECFP_PC42", "ECFP_PC43", "ECFP_PC44", "ECFP_PC45", "ECFP_PC46", "ECFP_PC47", "ECFP_PC48", 
        "ECFP_PC49", "ECFP_PC50", 
        "Mordred_PC1", 
        "ChemBERTa_PC1", "ChemBERTa_PC2", "ChemBERTa_PC3", "ChemBERTa_PC4", "ChemBERTa_PC5", 
        "ChemBERTa_PC6", "ChemBERTa_PC7", "ChemBERTa_PC8", "ChemBERTa_PC9", "ChemBERTa_PC10", "ChemBERTa_PC11", 
        "ChemBERTa_PC12", "ChemBERTa_PC13", "ChemBERTa_PC14", "ChemBERTa_PC15", "ChemBERTa_PC16", "ChemBERTa_PC17", 
        "ChemBERTa_PC18", "ChemBERTa_PC19", "ChemBERTa_PC20", "ChemBERTa_PC21", "ChemBERTa_PC22", "ChemBERTa_PC23", 
        "ChemBERTa_PC24", "ChemBERTa_PC25", "ChemBERTa_PC26", "ChemBERTa_PC27", "ChemBERTa_PC28", "ChemBERTa_PC29", 
        "ChemBERTa_PC30", "ChemBERTa_PC31", "ChemBERTa_PC32", "ChemBERTa_PC33", "ChemBERTa_PC34", "ChemBERTa_PC35", 
        "ChemBERTa_PC36"
    ]
    
    # 分析描述符类型
    descriptor_analysis = []
    
    # 1. 基础信息列（非描述符）
    basic_info = ["Name", "Empirical formula", "Canonical SMILES", "log_EC50"]
    descriptor_analysis.append({
        "描述符类型": "基础信息列（非描述符）",
        "起始列": 1,
        "结束列": 4,
        "列数": 4,
        "包含列": "Name, Empirical formula, Canonical SMILES, log_EC50",
        "说明": "这些是基本信息，不是分子描述符"
    })
    
    # 2. RDKit基础物理化学描述符
    rdkit_start = 5
    rdkit_end = 43
    rdkit_columns = columns[rdkit_start-1:rdkit_end]
    descriptor_analysis.append({
        "描述符类型": "RDKit基础物理化学描述符",
        "起始列": rdkit_start,
        "结束列": rdkit_end,
        "列数": rdkit_end - rdkit_start + 1,
        "包含列": ", ".join(rdkit_columns),
        "说明": "基础物理化学描述符（分子量、LogP、氢键供体受体、拓扑描述符、分子片段等）"
    })
    
    # 3. MACCS密钥指纹
    maccs_start = 44
    maccs_end = 75
    maccs_columns = columns[maccs_start-1:maccs_end]
    descriptor_analysis.append({
        "描述符类型": "MACCS密钥指纹",
        "起始列": maccs_start,
        "结束列": maccs_end,
        "列数": maccs_end - maccs_start + 1,
        "包含列": f"MACCS_PC1 到 MACCS_PC32（共32个主成分）",
        "说明": "MACCS密钥指纹（166位）的主成分分析结果"
    })
    
    # 4. ECFP扩展连通性指纹
    ecfp_start = 76
    ecfp_end = 125
    ecfp_columns = columns[ecfp_start-1:ecfp_end]
    descriptor_analysis.append({
        "描述符类型": "ECFP扩展连通性指纹",
        "起始列": ecfp_start,
        "结束列": ecfp_end,
        "列数": ecfp_end - ecfp_start + 1,
        "包含列": f"ECFP_PC1 到 ECFP_PC50（共50个主成分）",
        "说明": "扩展连通性指纹（ECFP4，2048位）的主成分分析结果"
    })
    
    # 5. Mordred描述符
    mordred_start = 126
    mordred_end = 126
    mordred_columns = columns[mordred_start-1:mordred_end]
    descriptor_analysis.append({
        "描述符类型": "Mordred描述符",
        "起始列": mordred_start,
        "结束列": mordred_end,
        "列数": mordred_end - mordred_start + 1,
        "包含列": ", ".join(mordred_columns),
        "说明": "类似MOE的2D描述符（1300+种）的主成分分析结果"
    })
    
    # 6. ChemBERTa描述符
    chemberta_start = 127
    chemberta_end = 162
    chemberta_columns = columns[chemberta_start-1:chemberta_end]
    descriptor_analysis.append({
        "描述符类型": "ChemBERTa描述符",
        "起始列": chemberta_start,
        "结束列": chemberta_end,
        "列数": chemberta_end - chemberta_start + 1,
        "包含列": f"ChemBERTa_PC1 到 ChemBERTa_PC36（共36个主成分）",
        "说明": "基于Transformer的分子表示的主成分分析结果"
    })
    
    # 创建DataFrame
    df_analysis = pd.DataFrame(descriptor_analysis)
    
    # 保存到Excel
    with pd.ExcelWriter('分子描述符分析结果.xlsx') as writer:
        df_analysis.to_excel(writer, sheet_name='描述符类型分析', index=False)
        
        # 创建详细列名对照表
        detailed_columns = []
        for i, col in enumerate(columns, 1):
            col_type = ""
            if i <= 4:
                col_type = "基础信息列"
            elif i <= 43:
                col_type = "RDKit描述符"
            elif i <= 75:
                col_type = "MACCS指纹"
            elif i <= 125:
                col_type = "ECFP指纹"
            elif i == 126:
                col_type = "Mordred描述符"
            elif i <= 162:
                col_type = "ChemBERTa描述符"
            
            detailed_columns.append({
                "列序号": i,
                "列名": col,
                "描述符类型": col_type
            })
        
        detailed_df = pd.DataFrame(detailed_columns)
        detailed_df.to_excel(writer, sheet_name='详细列名对照', index=False)
        
        # 创建统计摘要
        summary_data = []
        for desc_type in df_analysis['描述符类型'].unique():
            if desc_type != "基础信息列（非描述符）":
                count = df_analysis[df_analysis['描述符类型'] == desc_type]['列数'].iloc[0]
                summary_data.append({
                    "描述符类型": desc_type,
                    "特征数量": count,
                    "占总特征比例": f"{count/158*100:.1f}%"
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
    
    print("分子描述符分析完成！结果已保存到 '分子描述符分析结果.xlsx'")
    print("\n=== 分析结果摘要 ===")
    print(df_analysis.to_string(index=False))
    
    return df_analysis

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

def create_feature_subsets(X, feature_cols):
    """创建三种消融实验的特征子集"""
    feature_subsets = {}
    
    # 1. 完整特征集（所有描述符）
    feature_subsets['Full_Features'] = feature_cols
    
    # 2. 去掉Mordred和ChemBERTa
    no_mordred_chemberta = [col for col in feature_cols 
                           if not (col.startswith('Mordred_') or col.startswith('ChemBERTa_'))]
    feature_subsets['RDKit+ Fingerprint'] = no_mordred_chemberta
    
    # 3. 只用RDKit描述符
    rdkit_only = [col for col in feature_cols 
                  if col in ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "HeavyAtomCount", 
                           "RingCount", "BalabanJ", "BertzCT", "Chi0n", "Chi1n", "Chi2n", "Kappa1", 
                           "Kappa2", "HallKierAlpha", "FractionCSP3", "NumRadicalElectrons", 
                           "NumValenceElectrons", "NumAmideBonds", "NumAromaticCarbocycles", 
                           "NumAromaticHeterocycles", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", 
                           "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "LipinskiHBA", 
                           "LipinskiHBD", "LipinskiRuleOf5Failures", "SLogP", "LabuteASA", "PEOE_VSA1", 
                           "fr_Al_OH", "fr_Ar_OH", "fr_COO", "fr_NH0", "fr_NH1", "fr_NH2", "fr_SH", "fr_halogen"]]
    feature_subsets['RDKit_Only'] = rdkit_only
    
    return feature_subsets

def train_and_tune_rf(X_train, X_test, y_train, y_test, feature_subset, subset_name):
    """训练和调优随机森林模型（简化版本，去掉交叉验证）"""
    logger.info(f"\n开始训练和调优 {subset_name} 模型...")
    logger.info(f"特征数量: {len(feature_subset)}")
    
    # 选择特征子集
    X_train_subset = X_train[feature_subset]
    X_test_subset = X_test[feature_subset]
    
    # 默认模型
    default_rf = RandomForestRegressor(random_state=42)
    default_rf.fit(X_train_subset, y_train)
    y_pred_default = default_rf.predict(X_test_subset)
    default_metrics = {
        'MSE': mean_squared_error(y_test, y_pred_default),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_default)),
        'MAE': mean_absolute_error(y_test, y_pred_default),
        'R2': r2_score(y_test, y_pred_default),
        'MRE': calculate_mre(y_test, y_pred_default)
    }
    
    # 记录默认参数
    default_params = default_rf.get_params()
    
    # 简化的参数网格（去掉交叉验证）
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    logger.info("开始网格搜索...")
    start_time = time.time()
    
    # 手动网格搜索（去掉交叉验证）
    best_score = -float('inf')
    best_params = None
    best_model = None
    
    total_combinations = 1
    for param in param_grid.values():
        total_combinations *= len(param)
    
    logger.info(f"总共需要测试 {total_combinations} 种参数组合")
    
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_samples_leaf in param_grid['min_samples_leaf']:
                    for max_features in param_grid['max_features']:
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'max_features': max_features,
                            'random_state': 42
                        }
                        
                        # 训练模型
                        rf = RandomForestRegressor(**params)
                        rf.fit(X_train_subset, y_train)
                        
                        # 在测试集上评估
                        y_pred = rf.predict(X_test_subset)
                        score = r2_score(y_test, y_pred)
                        
                        # 更新最佳参数
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                            best_model = rf
    
    logger.info(f"网格搜索完成，耗时: {time.time() - start_time:.2f}秒")
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳测试集R²: {best_score:.4f}")
    
    # 最佳模型
    y_pred_best = best_model.predict(X_test_subset)
    best_metrics = {
        'MSE': mean_squared_error(y_test, y_pred_best),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_best)),
        'MAE': mean_absolute_error(y_test, y_pred_best),
        'R2': r2_score(y_test, y_pred_best),
        'MRE': calculate_mre(y_test, y_pred_best)
    }
    
    # 记录最佳参数
    best_full_params = best_model.get_params()
    
    logger.info(f"{subset_name} 模型训练和调优完成")
    logger.info(f"默认参数 - 测试集 MAE: {default_metrics['MAE']:.4f}, R²: {default_metrics['R2']:.4f}, MRE: {default_metrics['MRE']:.2f}%")
    logger.info(f"调优后 - 测试集 MAE: {best_metrics['MAE']:.4f}, R²: {best_metrics['R2']:.4f}, MRE: {best_metrics['MRE']:.2f}%")
    
    return {
        'default_metrics': default_metrics,
        'best_metrics': best_metrics,
        'best_params': best_params,
        'param_grid': param_grid,
        'cv_results': None,  # 去掉交叉验证结果
        'default_params': default_params,
        'best_full_params': best_full_params,
        'model': best_model,
        'feature_subset': feature_subset
    }

def plot_ablation_radar_chart(ablation_results, output_dir):
    """绘制消融实验雷达图 - 修复版本"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 指标名称（包含MRE）
    metrics = ['R²', 'MAE', 'MSE', 'RMSE', 'MRE']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    # 定义美观的颜色方案
    colors = ['#FF6B6B', '#4ECDC4', '#96CEB4']  # 红色、青色、绿色
    markers = ['s', 'o', '^']  # 方形、圆形、三角形
    
    # 为每个模型创建雷达图（使用调优后的最佳指标）
    for i, (model_name, result) in enumerate(ablation_results.items()):
        best_metrics = result['best_metrics']
        
        # 归一化指标值（对于误差指标，值越小越好；对于R2，值越大越好）
        normalized_values = []
        for metric in metrics:
            if metric == 'R²':
                # R²：直接归一化（值越大越好）
                normalized_values.append(best_metrics['R2'])
            else:
                # 误差指标：1 - 归一化值（值越小越好）
                if metric == 'MAE':
                    max_val = max([r['best_metrics']['MAE'] for r in ablation_results.values()]) * 1.1
                    normalized_values.append(1 - (best_metrics['MAE'] / max_val))
                elif metric == 'MSE':
                    max_val = max([r['best_metrics']['MSE'] for r in ablation_results.values()]) * 1.1
                    normalized_values.append(1 - (best_metrics['MSE'] / max_val))
                elif metric == 'RMSE':
                    max_val = max([r['best_metrics']['RMSE'] for r in ablation_results.values()]) * 1.1
                    normalized_values.append(1 - (best_metrics['RMSE'] / max_val))
                elif metric == 'MRE':
                    max_val = max([r['best_metrics']['MRE'] for r in ablation_results.values()]) * 1.1
                    normalized_values.append(1 - (best_metrics['MRE'] / max_val))
        
        # 闭合雷达图
        normalized_values.append(normalized_values[0])
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles.append(angles[0])
        
        # 绘制雷达图
        ax.plot(angles, normalized_values, linewidth=2.5, markersize=6, 
                color=colors[i], marker=markers[i], label=model_name, alpha=0.9)
        ax.fill(angles, normalized_values, alpha=0.1, color=colors[i])
    
    # 设置角度和标签
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=10, fontweight='bold')
    
    # 设置雷达图范围
    ax.set_ylim(0, 1.2)
    
    # 设置径向网格线
    ax.set_rticks([0.0, 0.4, 0.8, 1.2])
    ax.set_rlabel_position(0)
    
    # 移除背景网格线
    ax.grid(False)
    
    # 设置背景
    ax.set_facecolor('#E8F4F8')  # 浅蓝绿色背景
    fig.patch.set_facecolor('white')
    
    # 添加图例（修复fontweight参数问题）
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, 
              frameon=True, fancybox=True, shadow=True, framealpha=0.9,
              title='RF Model', title_fontsize=11)
    
    # 添加标题
    plt.title('(b)', fontsize=12, fontweight='bold', pad=20, loc='left')
    
    # 在右上角添加模型标识
    ax.text(0.95, 0.95, 'RF Model', transform=ax.transAxes, fontsize=14, 
            fontweight='bold', verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                     edgecolor='gray', linewidth=1))
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, 'ablation_radar_chart.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'ablation_radar_chart.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"消融实验雷达图已保存至: {output_dir}")

def save_ablation_results(ablation_results, output_dir):
    """保存消融实验结果（只保存模型性能比较）"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果汇总表
    summary_data = []
    for model_name, result in ablation_results.items():
        default_metrics = result['default_metrics']
        best_metrics = result['best_metrics']
        
        summary_data.append({
            'Model': model_name,
            'Features_Count': len(result['feature_subset']),
            'Default_MAE': default_metrics['MAE'],
            'Default_R2': default_metrics['R2'],
            'Default_MRE': default_metrics['MRE'],
            'Tuned_MAE': best_metrics['MAE'],
            'Tuned_R2': best_metrics['R2'],
            'Tuned_MRE': best_metrics['MRE'],
            'Improvement_MAE': default_metrics['MAE'] - best_metrics['MAE'],
            'Improvement_R2': best_metrics['R2'] - default_metrics['R2'],
            'Improvement_MRE': default_metrics['MRE'] - best_metrics['MRE']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存到Excel（只保存性能比较结果）
    with pd.ExcelWriter(os.path.join(output_dir, 'ablation_study_results.xlsx')) as writer:
        summary_df.to_excel(writer, sheet_name='Performance_Comparison', index=False)
        
        # 保存每个模型的详细性能指标
        performance_details = []
        for model_name, result in ablation_results.items():
            default_metrics = result['default_metrics']
            best_metrics = result['best_metrics']
            
            performance_details.append({
                'Model': model_name,
                'Metric': 'Default',
                'MSE': default_metrics['MSE'],
                'RMSE': default_metrics['RMSE'],
                'MAE': default_metrics['MAE'],
                'R2': default_metrics['R2'],
                'MRE': default_metrics['MRE']
            })
            
            performance_details.append({
                'Model': model_name,
                'Metric': 'Tuned',
                'MSE': best_metrics['MSE'],
                'RMSE': best_metrics['RMSE'],
                'MAE': best_metrics['MAE'],
                'R2': best_metrics['R2'],
                'MRE': best_metrics['MRE']
            })
        
        performance_df = pd.DataFrame(performance_details)
        performance_df.to_excel(writer, sheet_name='Detailed_Performance', index=False)
    
    logger.info(f"消融实验结果已保存至: {output_dir}")

def main():
    """主函数"""
    try:
        # 1. 分析描述符类型
        logger.info("开始分析分子描述符类型...")
        analyze_descriptor_types()
        
        # 2. 加载数据
        input_file = "MCF-7_molecular_descriptors_reduced1.xlsx"
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info(f"开始处理文件: {input_file}")
        start_time = time.time()
        
        # 加载和准备数据
        X_train, X_test, y_train, y_test, feature_cols, df = load_and_prepare_data(input_file)
        
        # 3. 创建特征子集
        feature_subsets = create_feature_subsets(X_train, feature_cols)
        
        # 4. 执行消融实验（简化版本）
        ablation_results = {}
        for subset_name, feature_subset in feature_subsets.items():
            result = train_and_tune_rf(X_train, X_test, y_train, y_test, feature_subset, subset_name)
            ablation_results[subset_name] = result
        
        # 5. 绘制雷达图
        output_dir = './ablation_study_results'
        plot_ablation_radar_chart(ablation_results, output_dir)
        
        # 6. 保存结果（只保存性能比较）
        save_ablation_results(ablation_results, output_dir)
        
        # 7. 保存模型
        for model_name, result in ablation_results.items():
            model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
            joblib.dump(result['model'], model_path)
        
        # 记录执行时间
        elapsed = time.time() - start_time
        logger.info(f"\n总执行时间: {elapsed:.2f}秒")
        logger.info("消融实验完成!")
        
        # 打印最终结果
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS SUMMARY")
        print("="*80)
        for model_name, result in ablation_results.items():
            default_metrics = result['default_metrics']
            best_metrics = result['best_metrics']
            print(f"{model_name}:")
            print(f"  Features: {len(result['feature_subset'])}")
            print(f"  Default - MAE: {default_metrics['MAE']:.4f}, R²: {default_metrics['R2']:.4f}, MRE: {default_metrics['MRE']:.2f}%")
            print(f"  Tuned - MAE: {best_metrics['MAE']:.4f}, R²: {best_metrics['R2']:.4f}, MRE: {best_metrics['MRE']:.2f}%")
            print(f"  Improvement - MAE: {default_metrics['MAE'] - best_metrics['MAE']:.4f}, R²: {best_metrics['R2'] - default_metrics['R2']:.4f}, MRE: {default_metrics['MRE'] - best_metrics['MRE']:.2f}%")
            print()
        
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise

if __name__ == "__main__":
    main()