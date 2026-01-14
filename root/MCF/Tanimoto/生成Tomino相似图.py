import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

# 设置科学期刊绘图样式
plt.style.use('default')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['xtick.direction'] = 'out'  # 修改为刻度朝外
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.major.size'] = 3  # 减小刻度大小
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 600

# 全局字体设置
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

# 读取数据、生成指纹和相似度矩阵部分保持不变...
df = pd.read_excel('MCF-7_fine_tune_data.xlsx')
smiles_list = df['Canonical SMILES'].tolist()

mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols if mol is not None]

n = len(fps)
similarity_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])

il_labels = [f'IL{i+1}' for i in range(n)]

# 定义统一的绘图区域大小（相对坐标）和图像尺寸
figsize_inch = 17 / 2.54  # 17 cm → inch
plot_area_left = 0.18
plot_area_bottom = 0.15
plot_area_width = 0.65  # 缩小宽度给颜色条留出空间
plot_area_height = 0.70
position = [plot_area_left, plot_area_bottom, plot_area_width, plot_area_height]

# 图1: 热力图（正方形，统一绘图区域）
colors = ['#ffffcc', '#e6f598', '#c6e784', '#a5d96b', '#84cc5a', '#63be49', '#45b037', '#2c9e25']
cmap = mpl.colors.LinearSegmentedColormap.from_list('bright_yellow_green', colors, N=100)

fig1, ax1 = plt.subplots(figsize=(figsize_inch, figsize_inch))
sns.heatmap(similarity_matrix,
            cmap=cmap,
            vmin=0.0, vmax=1.0,
            square=True,
            ax=ax1,
            cbar_kws={'shrink': 0.6})

step = 50
if n > step:
    ticks = np.arange(0, n, step)
    tick_labels = [il_labels[i] for i in ticks]
    ax1.set_xticks(ticks + 0.5)
    ax1.set_yticks(ticks + 0.5)
    ax1.set_xticklabels(tick_labels, rotation=90, weight='bold', color='black')
    ax1.set_yticklabels(tick_labels, rotation=0, weight='bold', color='black')
else:
    ax1.set_xticks(np.arange(n) + 0.5)
    ax1.set_yticks(np.arange(n) + 0.5)
    ax1.set_xticklabels(il_labels, rotation=90, weight='bold', color='black')
    ax1.set_yticklabels(il_labels, rotation=0, weight='bold', color='black')

ax1.tick_params(axis='both', which='both', direction=mpl.rcParams['xtick.direction'])
ax1.set_position(position)

# 颜色条设置
cbar = ax1.collections[0].colorbar
cbar.ax.yaxis.label.set_weight('bold')
cbar.ax.yaxis.label.set_size(12)
for label in cbar.ax.yaxis.get_ticklabels():
    label.set_weight('bold')
    label.set_color('black')
cbar.ax.yaxis.set_tick_params(pad=4)

plt.tight_layout()
plt.savefig('Tanimoto_heatmap.tif', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('Tanimoto_heatmap.pdf', bbox_inches='tight', facecolor='white')
plt.savefig('Tanimoto_heatmap.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# ✅ 关键修改：确保绘图区域是矩形，宽度稍大于高度
plot_area_width = 0.75  # 宽度增加
plot_area_height = 0.65  # 高度保持一致
position = [plot_area_left, plot_area_bottom, plot_area_width, plot_area_height]

fig2, ax2 = plt.subplots(figsize=(figsize_inch, figsize_inch))

non_diag_values = similarity_matrix[~np.eye(n, dtype=bool)].flatten()
hist, bin_edges = np.histogram(non_diag_values, bins=20, range=(0.0, 1.0))  # 将范围扩展至1.0
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
proportions = hist / hist.sum()

bright_blue = '#3d85c6'
ax2.bar(bin_centers, proportions, width=0.035, alpha=0.80, color=bright_blue, edgecolor='white', linewidth=0.5)
ax2.plot(bin_centers, proportions, 'o-', color='#ff7f0e', linewidth=2, markersize=5,
         markerfacecolor='white', markeredgecolor='#ff7f0e', markeredgewidth=1.5)

ax2.set_xlabel('Tanimoto Coefficient', fontsize=12, fontfamily='Times New Roman', labelpad=10, weight='bold', color='black')
ax2.set_ylabel('Proportion', fontsize=12, fontfamily='Times New Roman', labelpad=10, weight='bold', color='black')
ax2.set_xlim(0.0, 1.1)  # x轴范围扩展至1.0
ax2.set_ylim(0.0, max(proportions) * 1.1 if max(proportions) < 1 else 1.0)  # 根据数据动态设置y轴上限

ax2.tick_params(axis='both', which='both', direction=mpl.rcParams['xtick.direction'])

# 使用 FormatStrFormatter 保证所有刻度显示两位小数
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 刻度标签加粗和颜色
for label in ax2.get_xticklabels():
    label.set_weight('bold')
    label.set_color('black')
for label in ax2.get_yticklabels():
    label.set_weight('bold')
    label.set_color('black')

# 隐藏 x 和 y 轴上的第一个刻度标签（即 0.00）
ax2.xaxis.get_major_ticks()[0].label1.set_visible(False)
ax2.yaxis.get_major_ticks()[0].label1.set_visible(False)

# ✅ 手动添加一个 0.00，可微调位置
ax2.text(-0.01, -0.01, '0.00', transform=ax2.transAxes, fontsize=12,
         va='top', ha='right', weight='bold', color='black')

# ✅ 设置矩形绘图区域，宽度稍大于高度
ax2.set_position(position)

# 保存图像
plt.savefig('Tanimoto_distribution.tif', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('Tanimoto_distribution.pdf', bbox_inches='tight', facecolor='white')
plt.savefig('Tanimoto_distribution.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()


# 计算并输出整体相似度和标准差
overall_similarity = np.mean(similarity_matrix[~np.eye(n, dtype=bool)])
std_similarity = np.std(similarity_matrix[~np.eye(n, dtype=bool)])

print(f"Overall Similarity: {overall_similarity:.4f}")
print(f"Standard Deviation of Similarity: {std_similarity:.4f}")