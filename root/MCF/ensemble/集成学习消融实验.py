import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
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
        logging.FileHandler("svr_ensemble_ablation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_descriptor_types():
    """分析描述符类型并输出到Excel"""
    # 分析您的数据列
    columns = [
        "Name", "Empirical formula", "Canonical SMILES", "log_EC50", 
        "MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "HeavyAtomCount", "RingCount", 
        "BalabanJ", "BertzCT", "Chi0n", "Chi1n", "Chi2n", "Kappa1", "Kappa2", "HallKierAlpha", 
        "FractionCSP3", "NumRadicalElectrons", "NumValenceElectrons", "NumAmideBonds", 
        "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAliphaticCarbocycles", 
        "NumAliphaticHeterocycles", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", 
        "LipinskiHBA", "LipinskiHBD", "LipinskiRuleOf5Failures", "SLogP", "LabuteASA", 
        "PEOE_VSA1", "fr_Al_OH", "fr_Ar_OH", "fr_COO", "fr_NH0", "fr_NH1", "fr_NH2", 
        "fr_SH", "fr_halogen", 
        "MACCS_PC1", "MACCS_PC2", "MACCS_PC3", "MACCS_PC4", "MACCS_PC5", "MACCS_PC6", 
        "MACCS_PC7", "MACCS_PC8", "MACCS_PC9", "MACCS_PC10", "MACCS_PC11", "MACCS_PC12", 
        "MACCS_PC13", "MACCS_PC14", "MACCS_PC15", "MACCS_PC16", "MACCS_PC17", "MACCS_PC18", 
        "MACCS_PC19", "MACCS_PC20", "MACCS_PC21", "MACCS_PC22", "MACCS_PC23", "MACCS_PC24", 
        "MACCS_PC25", "MACCS_PC26", "MACCS_PC27", "MACCS_PC28", "MACCS_PC29", "MACCS_PC30", 
        "MACCS_PC31", "MACCS_PC32", 
        "ECFP_PC1", "ECFP_PC2", "ECFP_PC3", "ECFP_PC4", "ECFP_PC5", "ECFP_PC6", "ECFP_PC7", 
        "ECFP_PC8", "ECFP_PC9", "ECFP_PC10", "ECFP_PC11", "ECFP_PC12", "ECFP_PC13", "ECFP_PC14", 
        "ECFP_PC15", "ECFP_PC16", "ECFP_PC17", "ECFP_PC18", "ECFP_PC19", "ECFP_PC20", "ECFP_PC21", 
        "ECFP_PC22", "ECFP_PC23", "ECFP_PC24", "ECFP_PC25", "ECFP_PC26", "ECFP_PC27", "ECFP_PC28", 
        "ECFP_PC29", "ECFP_PC30", "ECFP_PC31", "ECFP_PC32", "ECFP_PC33", "ECFP_PC34", "ECFP_PC35", 
        "ECFP_PC36", "ECFP_PC37", "ECFP_PC38", "ECFP_PC39", "ECFP_PC40", "ECFP_PC41", "ECFP_PC42", 
        "ECFP_PC43", "ECFP_PC44", "ECFP_PC45", "ECFP_PC46", "ECFP_PC47", "ECFP_PC48", "ECFP_PC49", 
        "ECFP_PC50", 
        "Mordred_PC1", "ChemBERTa_PC1", "ChemBERTa_PC2", "ChemBERTa_PC3", "ChemBERTa_PC4", 
        "ChemBERTa_PC5", "ChemBERTa_PC6", "ChemBERTa_PC7", "ChemBERTa_PC8", "ChemBERTa_PC9", 
        "ChemBERTa_PC10", "ChemBERTa_PC11", "ChemBERTa_PC12", "ChemBERTa_PC13", "ChemBERTa_PC14", 
        "ChemBERTa_PC15", "ChemBERTa_PC16", "ChemBERTa_PC17", "ChemBERTa_PC18", "ChemBERTa_PC19", 
        "ChemBERTa_PC20", "ChemBERTa_PC21", "ChemBERTa_PC22", "ChemBERTa_PC23", "ChemBERTa_PC24", 
        "ChemBERTa_PC25", "ChemBERTa_PC26", "ChemBERTa_PC27", "ChemBERTa_PC28", "ChemBERTa_PC29", 
        "ChemBERTa_PC30", "ChemBERTa_PC31", "ChemBERTa_PC32", "ChemBERTa_PC33", "ChemBERTa_PC34", 
        "ChemBERTa_PC35", "ChemBERTa_PC36"
    ]
    
    # 定义描述符类型
    descriptor_types = {
        "RDKit": "基础物理化学描述符（分子量、LogP等）",
        "Mordred": "类似MOE的2D描述符（1300+种）",
        "MACCS": "MACCS密钥指纹（166位）",
        "ECFP": "扩展连通性指纹（ECFP4，2048位）",
        "ChemBERTa": "基于Transformer的分子表示"
    }
    
    # 分析每列属于哪个类型
    analysis_results = []
    
    for i, col in enumerate(columns):
        if col in ["Name", "Empirical formula", "Canonical SMILES", "log_EC50"]:
            col_type = "标识列"
            description = "分子标识和目标变量"
        elif col in ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "HeavyAtomCount", 
                     "RingCount", "BalabanJ", "BertzCT", "Chi0n", "Chi1n", "Chi2n", "Kappa1", 
                     "Kappa2", "HallKierAlpha", "FractionCSP3", "NumRadicalElectrons", 
                     "NumValenceElectrons", "NumAmideBonds", "NumAromaticCarbocycles", 
                     "NumAromaticHeterocycles", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", 
                     "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "LipinskiHBA", 
                     "LipinskiHBD", "LipinskiRuleOf5Failures", "SLogP", "LabuteASA", "PEOE_VSA1", 
                     "fr_Al_OH", "fr_Ar_OH", "fr_COO", "fr_NH0", "fr_NH1", "fr_NH2", "fr_SH", "fr_halogen"]:
            col_type = "RDKit"
            description = descriptor_types["RDKit"]
        elif col.startswith("MACCS_PC"):
            col_type = "MACCS"
            description = descriptor_types["MACCS"]
        elif col.startswith("ECFP_PC"):
            col_type = "ECFP"
            description = descriptor_types["ECFP"]
        elif col.startswith("Mordred_PC"):
            col_type = "Mordred"
            description = descriptor_types["Mordred"]
        elif col.startswith("ChemBERTa_PC"):
            col_type = "ChemBERTa"
            description = descriptor_types["ChemBERTa"]
        else:
            col_type = "未知"
            description = "未分类的描述符"
        
        analysis_results.append({
            "列序号": i + 1,
            "列名": col,
            "描述符类型": col_type,
            "描述": description
        })
    
    # 创建汇总表
    summary_data = []
    for desc_type in descriptor_types.keys():
        count = len([r for r in analysis_results if r["描述符类型"] == desc_type])
        if count > 0:
            summary_data.append({
                "描述符类型": desc_type,
                "描述": descriptor_types[desc_type],
                "列数": count,
                "起始列": min([r["列序号"] for r in analysis_results if r["描述符类型"] == desc_type]),
                "结束列": max([r["列序号"] for r in analysis_results if r["描述符类型"] == desc_type])
            })
    
    # 保存到Excel
    with pd.ExcelWriter('descriptor_analysis.xlsx') as writer:
        pd.DataFrame(analysis_results).to_excel(writer, sheet_name='详细分析', index=False)
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='类型汇总', index=False)
    
    logger.info("描述符分析结果已保存到 descriptor_analysis.xlsx")
    return summary_data

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

def create_feature_subsets(feature_cols):
    """创建三种消融实验的特征子集"""
    # 1. 完整特征集
    full_features = feature_cols
    
    # 2. 去掉Mordred和ChemBERTa
    no_mordred_chemberta = [col for col in feature_cols 
                           if not (col.startswith('Mordred_PC') or col.startswith('ChemBERTa_PC'))]
    
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
    
    return {
        '完整特征': full_features,
        '无Mordred+ChemBERTa': no_mordred_chemberta,
        '仅RDKit描述符': rdkit_only
    }

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

def optimize_ensemble_model(X_train, y_train, feature_cols):
    """优化集成模型 - 与原始代码完全一致的优化方法"""
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
    
    # 优化集成权重 - 使用您的最佳权重0.6和0.4
    best_ensemble = VotingRegressor([
        ('svr', best_svr),
        ('enet', best_enet)
    ], weights=[0.6, 0.4])  # 使用您的最佳权重
    
    best_ensemble.fit(X_train, y_train)
    
    logger.info(f"SVR最佳参数: {svr_search.best_params_}")
    logger.info(f"弹性网络最佳参数: {enet_search.best_params_}")
    logger.info(f"集成权重: SVR=0.6, ElasticNet=0.4")
    
    return best_ensemble, svr_search, enet_search

def evaluate_model(y_true, y_pred):
    """评估模型性能 - 添加MRE指标，与原始代码一致"""
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

def train_ablation_model(X_train, X_test, y_train, y_test, feature_subset, subset_name):
    """训练消融实验模型 - 使用与原始代码完全一致的方法"""
    logger.info(f"开始训练 {subset_name} 消融实验模型...")
    
    # 使用特征子集
    X_train_subset = X_train[feature_subset]
    X_test_subset = X_test[feature_subset]
    
    # 优化集成模型 - 使用与原始代码完全一致的方法
    ensemble_model, svr_search, enet_search = optimize_ensemble_model(
        X_train_subset, y_train, feature_subset
    )
    
    # 评估集成模型性能
    y_train_pred = ensemble_model.predict(X_train_subset)
    train_metrics = evaluate_model(y_train, y_train_pred)
    
    y_test_pred = ensemble_model.predict(X_test_subset)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    # 获取个体模型性能
    best_svr = svr_search.best_estimator_
    best_enet = enet_search.best_estimator_
    
    y_train_pred_svr = best_svr.predict(X_train_subset)
    y_test_pred_svr = best_svr.predict(X_test_subset)
    y_train_pred_enet = best_enet.predict(X_train_subset)
    y_test_pred_enet = best_enet.predict(X_test_subset)
    
    individual_metrics = {
        'SVR': {
            'train': evaluate_model(y_train, y_train_pred_svr),
            'test': evaluate_model(y_test, y_test_pred_svr)
        },
        'ElasticNet': {
            'train': evaluate_model(y_train, y_train_pred_enet),
            'test': evaluate_model(y_test, y_test_pred_enet)
        }
    }
    
    logger.info(f"{subset_name} 消融实验模型训练完成")
    logger.info(f"  集成模型测试集R²: {test_metrics['R2']:.4f}")
    logger.info(f"  SVR测试集R²: {individual_metrics['SVR']['test']['R2']:.4f}")
    logger.info(f"  ElasticNet测试集R²: {individual_metrics['ElasticNet']['test']['R2']:.4f}")
    
    return {
        'ensemble_model': ensemble_model,
        'svr_model': best_svr,
        'enet_model': best_enet,
        'svr_search': svr_search,
        'enet_search': enet_search,
        'ensemble_metrics': {
            'train': train_metrics,
            'test': test_metrics
        },
        'individual_metrics': individual_metrics,
        'feature_count': len(feature_subset)
    }

def plot_ablation_radar_chart(ablation_results, output_dir):
    """绘制消融实验雷达图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 指标名称 - 与evaluate_model返回的键名一致
    metrics = ['R²', 'MAE', 'MSE', 'RMSE', 'MRE (%)']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    # 定义美观的颜色方案
    colors = ['#FF6B6B', '#4ECDC4', '#96CEB4']  # 红色、青色、绿色
    
    # 为每个模型创建雷达图
    for i, (model_name, result) in enumerate(ablation_results.items()):
        metrics_values = result['ensemble_metrics']['test']
        
        # 归一化指标值
        normalized_values = []
        for metric in metrics:
            if metric == 'R²':
                normalized_values.append(metrics_values['R2'])
            else:
                # 使用正确的键名
                metric_key = metric if metric != 'MRE (%)' else 'MRE (%)'
                normalized_values.append(1 - min(metrics_values[metric_key] / 10, 0.9))
        
        # 闭合雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        normalized_values += normalized_values[:1]
        angles += angles[:1]
        
        # 绘制雷达图
        ax.plot(angles, normalized_values, 'o-', linewidth=2, 
               color=colors[i], alpha=0.8, label=model_name, markersize=6)
        ax.fill(angles, normalized_values, alpha=0.1, color=colors[i])
    
    # 设置雷达图样式
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=6, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 设置背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
             fontsize=6, frameon=True, fancybox=True, shadow=True)
    
    # 添加标题
    plt.title('SVR+ElasticNet集成模型消融实验性能比较', fontsize=10, fontweight='bold', pad=20)
    
    # 保存图片
    plt.savefig(f'{output_dir}/ablation_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/ablation_radar_chart.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("消融实验雷达图已保存")

def save_ablation_results(ablation_results, output_dir):
    """保存消融实验结果 - 与原始代码格式一致"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果汇总表
    results_summary = []
    for model_name, result in ablation_results.items():
        ensemble_metrics = result['ensemble_metrics']['test']
        summary_row = {
            '模型名称': model_name,
            '特征数量': result['feature_count'],
            '集成R²': f"{ensemble_metrics['R2']:.4f}",
            '集成MAE': f"{ensemble_metrics['MAE']:.4f}",
            '集成MSE': f"{ensemble_metrics['MSE']:.4f}",
            '集成RMSE': f"{ensemble_metrics['RMSE']:.4f}",
            '集成MRE(%)': f"{ensemble_metrics['MRE (%)']:.2f}",
            'SVR_R²': f"{result['individual_metrics']['SVR']['test']['R2']:.4f}",
            'ElasticNet_R²': f"{result['individual_metrics']['ElasticNet']['test']['R2']:.4f}",
            '集成权重': 'SVR=0.6, ElasticNet=0.4'
        }
        results_summary.append(summary_row)
    
    # 保存到Excel
    with pd.ExcelWriter(f'{output_dir}/ablation_results.xlsx') as writer:
        # 1. 消融实验结果汇总
        pd.DataFrame(results_summary).to_excel(writer, sheet_name='消融实验结果', index=False)
        
        # 2. 每个模型的详细性能指标
        for model_name, result in ablation_results.items():
            # 集成模型指标
            ensemble_metrics_df = pd.DataFrame({
                '指标': ['R²', 'MSE', 'RMSE', 'MAE', 'Explained Variance', 'MRE (%)'],
                '训练集': [
                    result['ensemble_metrics']['train']['R2'],
                    result['ensemble_metrics']['train']['MSE'],
                    result['ensemble_metrics']['train']['RMSE'],
                    result['ensemble_metrics']['train']['MAE'],
                    result['ensemble_metrics']['train']['Explained Variance'],
                    result['ensemble_metrics']['train']['MRE (%)']
                ],
                '测试集': [
                    result['ensemble_metrics']['test']['R2'],
                    result['ensemble_metrics']['test']['MSE'],
                    result['ensemble_metrics']['test']['RMSE'],
                    result['ensemble_metrics']['test']['MAE'],
                    result['ensemble_metrics']['test']['Explained Variance'],
                    result['ensemble_metrics']['test']['MRE (%)']
                ]
            })
            ensemble_metrics_df.to_excel(writer, sheet_name=f'{model_name}_集成模型指标', index=False)
            
            # 个体模型指标
            individual_metrics_df = pd.DataFrame({
                '模型': ['SVR', 'SVR', 'ElasticNet', 'ElasticNet'],
                '数据集': ['训练集', '测试集', '训练集', '测试集'],
                'R²': [
                    result['individual_metrics']['SVR']['train']['R2'],
                    result['individual_metrics']['SVR']['test']['R2'],
                    result['individual_metrics']['ElasticNet']['train']['R2'],
                    result['individual_metrics']['ElasticNet']['test']['R2']
                ],
                'RMSE': [
                    result['individual_metrics']['SVR']['train']['RMSE'],
                    result['individual_metrics']['SVR']['test']['RMSE'],
                    result['individual_metrics']['ElasticNet']['train']['RMSE'],
                    result['individual_metrics']['ElasticNet']['test']['RMSE']
                ]
            })
            individual_metrics_df.to_excel(writer, sheet_name=f'{model_name}_个体模型指标', index=False)
            
            # 最佳参数
            best_params_df = pd.DataFrame({
                '模型': ['SVR', 'ElasticNet'],
                '最佳参数': [str(result['svr_search'].best_params_), str(result['enet_search'].best_params_)]
            })
            best_params_df.to_excel(writer, sheet_name=f'{model_name}_最佳参数', index=False)
    
    logger.info("消融实验结果已保存到Excel")

def main():
    try:
        input_file = "MCF-7_molecular_descriptors_reduced1.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info("开始SVR+ElasticNet集成模型消融实验...")
        start_time = time.time()
        
        # 1. 分析描述符类型
        logger.info("分析描述符类型...")
        summary_data = analyze_descriptor_types()
        
        # 2. 加载和准备数据
        logger.info("加载和准备数据...")
        X_train, X_test, y_train, y_test, feature_cols, df = load_and_prepare_data(input_file)
        
        # 3. 创建特征子集
        logger.info("创建特征子集...")
        feature_subsets = create_feature_subsets(feature_cols)
        
        # 4. 训练三种消融实验模型
        ablation_results = {}
        for subset_name, subset_features in feature_subsets.items():
            result = train_ablation_model(X_train, X_test, y_train, y_test, subset_features, subset_name)
            ablation_results[subset_name] = result
        
        # 5. 绘制雷达图
        logger.info("绘制消融实验雷达图...")
        output_dir = 'svr_ensemble_ablation_results'
        plot_ablation_radar_chart(ablation_results, output_dir)
        
        # 6. 保存结果
        logger.info("保存消融实验结果...")
        save_ablation_results(ablation_results, output_dir)
        
        # 7. 保存模型
        for model_name, result in ablation_results.items():
            # 保存集成模型
            ensemble_model_path = f'{output_dir}/{model_name.replace(" ", "_")}_ensemble_model.pkl'
            joblib.dump(result['ensemble_model'], ensemble_model_path)
            
            # 保存个体模型
            svr_model_path = f'{output_dir}/{model_name.replace(" ", "_")}_svr_model.pkl'
            joblib.dump(result['svr_model'], svr_model_path)
            
            enet_model_path = f'{output_dir}/{model_name.replace(" ", "_")}_enet_model.pkl'
            joblib.dump(result['enet_model'], enet_model_path)
        
        elapsed = time.time() - start_time
        logger.info(f"消融实验完成，总耗时: {elapsed:.2f}秒")
        
        # 打印最终结果摘要
        logger.info("\n" + "="*50)
        logger.info("最终结果摘要")
        logger.info("="*50)
        for model_name, result in ablation_results.items():
            ensemble_metrics = result['ensemble_metrics']['test']
            individual_metrics = result['individual_metrics']
            
            logger.info(f"{model_name}:")
            logger.info(f"  特征数量: {result['feature_count']}")
            logger.info(f"  集成模型测试集R²: {ensemble_metrics['R2']:.4f}")
            logger.info(f"  集成模型测试集RMSE: {ensemble_metrics['RMSE']:.4f}")
            logger.info(f"  SVR测试集R²: {individual_metrics['SVR']['test']['R2']:.4f}")
            logger.info(f"  ElasticNet测试集R²: {individual_metrics['ElasticNet']['test']['R2']:.4f}")
            logger.info(f"  集成权重: SVR=0.6, ElasticNet=0.4")
            logger.info("-" * 30)
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
        logger.info("集成权重: SVR=0.6, ElasticNet=0.4")
        
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise

if __name__ == "__main__":
    main()