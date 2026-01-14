import os
import time
import logging
import numpy as np
import pandas as pd

from scipy.stats import uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import VotingRegressor

import joblib
import warnings
warnings.filterwarnings("ignore")

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ensemble_model_training.log", encoding="utf-8"),
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
        else:
            raise ValueError("FeatureSelector 目前仅支持 DataFrame 输入。")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices].values
        return X[:, self.selected_indices]

def load_and_prepare_data(file_path):
    """加载并划分数据"""
    df = pd.read_excel(file_path)
    logger.info(f"数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")

    target_col = "log_EC50"
    for name in ["log_EC50", "log EC50", "logEC50", "EC50_log", "log10(EC50)"]:
        if name in df.columns:
            target_col = name
            break

    non_feature_cols = [
        "Name", "Empirical formula", "Canonical SMILES", "SMILES", "smiles",
        "canonical_smiles", "log_EC50", "log EC50", "logEC50", "EC50_log", "log10(EC50)"
    ]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    if not feature_cols:
        raise ValueError("未找到特征列，请检查数据格式。")

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_cols

def create_pipelines(feature_cols):
    """构建两个单模型流水线"""
    svr_pipe = Pipeline([
        ("feature_selector", FeatureSelector(feature_cols)),
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])

    enet_pipe = Pipeline([
        ("feature_selector", FeatureSelector(feature_cols)),
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(max_iter=10000, tol=1e-4, random_state=42))
    ])
    return svr_pipe, enet_pipe

def tune_base_models(X_train, y_train, feature_cols):
    """对 SVR、ElasticNet 进行随机搜索"""
    svr_pipe, enet_pipe = create_pipelines(feature_cols)

    svr_params = {
        "svr__C": uniform(15, 5),
        "svr__gamma": uniform(0.001, 0.002),
        "svr__epsilon": [0.08, 0.1, 0.12]
    }
    enet_params = {
        "enet__alpha": uniform(0.03, 0.02),
        "enet__l1_ratio": uniform(0.35, 0.05),
        "enet__max_iter": [8000, 10000, 12000],
        "enet__tol": [8e-5, 1e-4, 1.2e-4]
    }

    logger.info("开始优化 SVR...")
    svr_search = RandomizedSearchCV(
        svr_pipe, svr_params, n_iter=30, cv=5,
        scoring="r2", n_jobs=-1, random_state=42, verbose=1
    )
    svr_search.fit(X_train, y_train)

    logger.info("开始优化 ElasticNet...")
    enet_search = RandomizedSearchCV(
        enet_pipe, enet_params, n_iter=30, cv=5,
        scoring="r2", n_jobs=-1, random_state=42, verbose=1
    )
    enet_search.fit(X_train, y_train)

    logger.info(f"SVR 最佳参数: {svr_search.best_params_}")
    logger.info(f"ElasticNet 最佳参数: {enet_search.best_params_}")

    return svr_search.best_estimator_, enet_search.best_estimator_, svr_search.best_params_, enet_search.best_params_

def evaluate_model(y_true, y_pred):
    """计算常用回归指标"""
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Explained Variance": explained_variance_score(y_true, y_pred)
    }

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

def build_ensemble(best_svr, best_enet, weights):
    """用给定权重构建 VotingRegressor"""
    return VotingRegressor(
        estimators=[("svr", best_svr), ("enet", best_enet)],
        weights=list(weights)
    )

def main():
    input_file = "MCF-7_molecular_descriptors_reduced1.xlsx"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件不存在: {input_file}")

    start_time = time.time()
    logger.info(f"开始处理: {input_file}")

    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(input_file)

    best_svr, best_enet, svr_params, enet_params = tune_base_models(
        X_train, y_train, feature_cols
    )

    # 训练单模型
    best_svr.fit(X_train, y_train)
    best_enet.fit(X_train, y_train)

    # 单模型指标
    svr_train_metrics = evaluate_model(y_train, best_svr.predict(X_train))
    svr_test_metrics = evaluate_model(y_test, best_svr.predict(X_test))
    enet_train_metrics = evaluate_model(y_train, best_enet.predict(X_train))
    enet_test_metrics = evaluate_model(y_test, best_enet.predict(X_test))

    logger.info("单模型测试集表现：")
    logger.info(f"SVR  -> R2={svr_test_metrics['R2']:.4f}, MAE={svr_test_metrics['MAE']:.4f}, MSE={svr_test_metrics['MSE']:.4f}, RMSE={svr_test_metrics['RMSE']:.4f}")
    logger.info(f"ENet -> R2={enet_test_metrics['R2']:.4f}, MAE={enet_test_metrics['MAE']:.4f}, MSE={enet_test_metrics['MSE']:.4f}, RMSE={enet_test_metrics['RMSE']:.4f}")

    # 四种权重方案
    metrics_order = ["R2", "MAE", "MSE", "RMSE"]
    ensemble_records = []

    for metric_name in metrics_order:
        weights = compute_weights(metric_name, svr_test_metrics, enet_test_metrics)
        ensemble_model = build_ensemble(best_svr, best_enet, weights)
        ensemble_model.fit(X_train, y_train)

        test_metrics = evaluate_model(y_test, ensemble_model.predict(X_test))
        train_metrics = evaluate_model(y_train, ensemble_model.predict(X_train))

        logger.info(f"权重方案（{metric_name}） -> 权重 {weights}, 测试集 R2={test_metrics['R2']:.4f}")
        ensemble_records.append({
            "Metric": metric_name,
            "SVR_Weight": weights[0],
            "ElasticNet_Weight": weights[1],
            "Test_R2": test_metrics["R2"],
            "Test_MAE": test_metrics["MAE"],
            "Test_MSE": test_metrics["MSE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Train_R2": train_metrics["R2"],
            "Train_MAE": train_metrics["MAE"],
            "Train_MSE": train_metrics["MSE"],
            "Train_RMSE": train_metrics["RMSE"]
        })

    # 保存结果到 Excel
    os.makedirs("ensemble_results", exist_ok=True)
    excel_path = "ensemble_results/ensemble_results2.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        base_metrics_df = pd.DataFrame({
            "Model": ["SVR", "SVR", "ElasticNet", "ElasticNet"],
            "Split": ["Train", "Test", "Train", "Test"],
            "R2": [svr_train_metrics["R2"], svr_test_metrics["R2"], enet_train_metrics["R2"], enet_test_metrics["R2"]],
            "MAE": [svr_train_metrics["MAE"], svr_test_metrics["MAE"], enet_train_metrics["MAE"], enet_test_metrics["MAE"]],
            "MSE": [svr_train_metrics["MSE"], svr_test_metrics["MSE"], enet_train_metrics["MSE"], enet_test_metrics["MSE"]],
            "RMSE": [svr_train_metrics["RMSE"], svr_test_metrics["RMSE"], enet_train_metrics["RMSE"], enet_test_metrics["RMSE"]],
            "Explained_Variance": [
                svr_train_metrics["Explained Variance"], svr_test_metrics["Explained Variance"],
                enet_train_metrics["Explained Variance"], enet_test_metrics["Explained Variance"]
            ]
        })
        base_metrics_df.to_excel(writer, sheet_name="BaseModelMetrics", index=False)

        ensemble_df = pd.DataFrame(ensemble_records)
        ensemble_df.to_excel(writer, sheet_name="EnsembleByMetric", index=False)

        params_df = pd.DataFrame([
            {"Model": "SVR", **svr_params},
            {"Model": "ElasticNet", **enet_params}
        ])
        params_df.to_excel(writer, sheet_name="BestParams", index=False)

    # 保存模型
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_svr, "saved_models/svr_best_model.pkl")
    joblib.dump(best_enet, "saved_models/enet_best_model.pkl")

    elapsed = time.time() - start_time
    logger.info(f"全部完成，耗时 {elapsed:.2f} 秒")
    logger.info(f"结果已保存：{excel_path}")

if __name__ == "__main__":
    main()