import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("svr_ablation.log", encoding='utf-8'),
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
        df = pd.read_excel(file_path)
        logger.info(f"数据加载成功: {df.shape[0]}行, {df.shape[1]}列")
        
        # 识别目标列和特征列
        target_col = 'log_EC50'
        non_feature_cols = ['Name', 'Empirical formula', 'Canonical SMILES', 'log_EC50']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        if not feature_cols:
            raise ValueError("未找到特征列！")
        
        # 分离特征和目标
        X = df[feature_cols]
        y = df[target_col]
        
        # 处理缺失值
        X = X.fillna(X.median())
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
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

def optimize_svr_model(X_train, y_train, feature_cols):
    """优化SVR模型（简化参数搜索）"""
    # SVR参数优化
    svr_pipe = Pipeline([
        ('feature_selector', FeatureSelector(feature_cols)),
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    
    # 简化的参数网格
    svr_params = {
        'svr__kernel': ['rbf', 'linear'],
        'svr__C': [0.1, 1, 10, 100],
        'svr__gamma': [0.001, 0.01, 0.1, 1],
        'svr__epsilon': [0.01, 0.1, 0.5]
    }
    
    logger.info("开始优化SVR模型...")
    svr_search = RandomizedSearchCV(
        svr_pipe, svr_params, n_iter=20, cv=3,  # 减少迭代次数和交叉验证折数
        scoring='r2', n_jobs=-1, verbose=1, random_state=42
    )
    svr_search.fit(X_train, y_train)
    
    logger.info(f"SVR最佳参数: {svr_search.best_params_}")
    logger.info(f"SVR最佳交叉验证R2: {svr_search.best_score_:.4f}")
    
    return svr_search.best_estimator_, svr_search

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MRE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics

def train_svr_model(X_train, X_test, y_train, y_test, feature_subset, subset_name):
    """训练SVR模型"""
    logger.info(f"开始训练 {subset_name} 模型...")
    
    # 使用特征子集
    X_train_subset = X_train[feature_subset]
    X_test_subset = X_test[feature_subset]
    
    # 修复：传递正确的参数
    svr_model, svr_search = optimize_svr_model(X_train_subset, y_train, feature_subset)
    
    # 评估模型
    y_pred = svr_model.predict(X_test_subset)
    metrics = evaluate_model(y_test, y_pred)
    
    logger.info(f"{subset_name} 模型训练完成，测试集R²: {metrics['R2']:.4f}")
    
    return {
        'model': svr_model,
        'search': svr_search,
        'best_params': svr_search.best_params_,
        'metrics': metrics,
        'feature_count': len(feature_subset)
    }

def plot_ablation_radar_chart(ablation_results, output_dir):
    """绘制消融实验雷达图 - 与参考图片完全一致的效果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 指标名称（使用中文，但字体调小）
    metrics = ['R²', 'MAE', 'MSE', 'RMSE', 'MRE']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    # 定义美观的颜色方案（与参考图片一致）
    colors = ['#FF6B6B', '#4ECDC4', '#96CEB4']  # 红色、青色、绿色
    markers = ['s', 'o', '^']  # 方形、圆形、三角形
    
    # 为每个模型创建雷达图
    for i, (model_name, result) in enumerate(ablation_results.items()):
        metrics_values = result['metrics']
        
        # 归一化指标值（对于误差指标，值越小越好；对于R²，值越大越好）
        normalized_values = []
        for metric in metrics:
            if metric == 'R²':
                # R²值越大越好，直接使用
                normalized_values.append(metrics_values['R2'])  # 使用英文键
            else:
                # 误差指标越小越好，需要反转
                normalized_values.append(1 - min(metrics_values[metric] / 10, 0.9))
        
        # 闭合雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        normalized_values += normalized_values[:1]  # 闭合
        angles += angles[:1]
        
        # 绘制雷达图
        ax.plot(angles, normalized_values, 'o-', linewidth=2, 
               color=colors[i], alpha=0.8, label=model_name, markersize=6)
        ax.fill(angles, normalized_values, alpha=0.1, color=colors[i])
    
    # 设置雷达图样式 - 字体调小避免编码问题
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=6, fontweight='bold')  # 字体从8改为6
    ax.set_ylim(0, 1)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 设置背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 添加图例（右上角）- 字体也调小
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
             fontsize=6, frameon=True, fancybox=True, shadow=True)  # 字体从8改为6
    
    # 添加标题 - 字体调小
    plt.title('SVR消融实验性能比较', fontsize=10, fontweight='bold', pad=20)  # 字体从12改为10
    
    # 保存图片
    plt.savefig(f'{output_dir}/ablation_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/ablation_radar_chart.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("消融实验雷达图已保存")

def save_ablation_results(ablation_results, output_dir):
    """保存消融实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果汇总表
    results_summary = []
    for model_name, result in ablation_results.items():
        summary_row = {
            '模型名称': model_name,
            '特征数量': result['feature_count'],
            'R²': f"{result['metrics']['R2']:.4f}",
            'MAE': f"{result['metrics']['MAE']:.4f}",
            'MSE': f"{result['metrics']['MSE']:.4f}",
            'RMSE': f"{result['metrics']['RMSE']:.4f}",
            'MRE(%)': f"{result['metrics']['MRE']:.2f}"
        }
        results_summary.append(summary_row)
    
    # 保存到Excel
    with pd.ExcelWriter(f'{output_dir}/ablation_results.xlsx') as writer:
        pd.DataFrame(results_summary).to_excel(writer, sheet_name='消融实验结果', index=False)
        
        # 保存每个模型的详细参数
        for model_name, result in ablation_results.items():
            params_df = pd.DataFrame([result['best_params']])
            params_df.to_excel(writer, sheet_name=f'{model_name}_参数', index=False)
    
    logger.info("消融实验结果已保存到Excel")

def main():
    try:
        input_file = "MCF-7_molecular_descriptors_reduced1.xlsx"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        
        logger.info("开始SVR消融实验...")
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
        
        # 4. 训练三种模型
        ablation_results = {}
        for subset_name, subset_features in feature_subsets.items():
            result = train_svr_model(X_train, X_test, y_train, y_test, subset_features, subset_name)
            ablation_results[subset_name] = result
        
        # 5. 绘制雷达图
        logger.info("绘制消融实验雷达图...")
        output_dir = 'svr_ablation_results'
        plot_ablation_radar_chart(ablation_results, output_dir)
        
        # 6. 保存结果
        logger.info("保存消融实验结果...")
        save_ablation_results(ablation_results, output_dir)
        
        # 7. 保存模型
        for model_name, result in ablation_results.items():
            model_path = f'{output_dir}/{model_name.replace(" ", "_")}_model.pkl'
            joblib.dump(result['model'], model_path)
        
        elapsed = time.time() - start_time
        logger.info(f"消融实验完成，总耗时: {elapsed:.2f}秒")
        
    except Exception as e:
        logger.error(f"程序终止: {str(e)}")
        raise

if __name__ == "__main__":
    main()