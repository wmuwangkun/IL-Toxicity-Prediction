# 离子液体毒性预测项目
基于机器学习的分子描述符预测研究，使用多种算法对Caco-2、IPC-81和MCF-7细胞系进行预测建模。

## 主要功能
**多种机器学习算法实现**：支持随机森林（RF）、多层感知机（MLP）、支持向量回归（SVR）、XGBoost、弹性网络（ElasticNet）等
- **集成学习**：使用投票回归器实现模型集成，提升预测性能
- **深度学习方法**：集成ChemBERTa预训练模型
- **模型评估与可视化**：提供详细的模型性能评估指标和可视化结果
- **消融实验**：分析不同特征子集对模型性能的影响
- **适用域分析**：评估模型预测的可靠性
- **特征重要性分析**：识别关键分子描述符

## 项目结构
├── Caco-2/ # Caco-2细胞系数据集
│ ├── ENet/ # 弹性网络模型
│ ├── MLP/ # 多层感知机模型
│ ├── RF/ # 随机森林模型
│ ├── SVR/ # 支持向量回归模型
│ ├── Tanimoto/ # Tanimoto相似度分析
│ ├── XGBoost/ # XGBoost模型
│ └── 集成学习/ # 集成学习模型
├── IPC-81/ # IPC-81细胞系数据集
│ └── [相同的子目录结构]
└── MCF/ # MCF-7细胞系数据集
└── [相同的子目录结构，包含消融实验]
└── ChemBERTa-77M-MLM/ # ChemBERTa预训练模型

## 环境要求
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
scipy>=1.7.0
openpyxl>=3.0.0### 安装依赖

##  使用方法
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib scipy openpyxl

## 将分子描述符数据文件（Excel格式）放置在对应数据集的目录中，例如：
- `Caco-2/CaCo-2_molecular_descriptors_reduced.xlsx`
- `IPC-81/IPC-81_molecular_descriptors_reduced.xlsx`
- `MCF/MCF-7_molecular_descriptors_reduced1.xlsx`

## 数据集说明
1. **Caco-2**：人结肠癌细胞系，常用于药物渗透性研究
2. **IPC-81**：人白血病细胞系
3. **MCF-7**：人乳腺癌细胞系

##  评估指标
- **MSE**（均方误差）：Mean Squared Error
- **RMSE**（均方根误差）：Root Mean Squared Error
- **MAE**（平均绝对误差）：Mean Absolute Error
- **R²**（决定系数）：Coefficient of Determination
