from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------
# 1. 加载数据 & 选择两个特征
# ----------------------------
iris = load_iris()
X = iris.data[:, [2, 3]]  # Petal Length (cm), Petal Width (cm)
y = iris.target           # 0: Setosa, 1: Versicolor, 2: Virginica
feature_names = ['Petal Length (cm)', 'Petal Width (cm)']
class_names = ['Setosa', 'Versicolor', 'Virginica']

# ----------------------------
# 2. 定义三种不同分类器
# ----------------------------
classifiers = {
    "Logistic Regression": LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=200
    ),
    "RBF SVM": SVC(kernel='rbf', probability=True, gamma=1, C=1),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42)
}

# ----------------------------
# 3. 创建网格用于可视化
# ----------------------------
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid = np.c_[xx.ravel(), yy.ravel()]

# ----------------------------
# 4. 绘图：每分类器 → 4 子图（1边界 + 3概率）
# ----------------------------
n_classifiers = len(classifiers)
fig, axes = plt.subplots(n_classifiers, 4, figsize=(16, 4 * n_classifiers))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 统一颜色：Setosa=red, Versicolor=green, Virginica=blue
colors = ['red', 'green', 'blue']
cmap_scatter = mcolors.ListedColormap(colors)
cmap_boundary = mcolors.ListedColormap(colors)

for idx, (name, clf) in enumerate(classifiers.items()):
    print(f"Training {name}...")
    clf.fit(X, y)
    
    # 预测类别 & 概率
    Z = clf.predict(grid).reshape(xx.shape)
    probs = clf.predict_proba(grid)  # shape: (n_grid, 3)
    probs = probs.reshape(xx.shape[0], xx.shape[1], 3)

    # --- 子图 0: 决策边界 ---
    ax = axes[idx, 0]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_boundary)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_scatter, 
                         edgecolors='k', s=40, linewidth=0.5)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(f'{name}\nDecision Boundary', fontsize=12)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    if idx == 0:
        ax.legend(handles=scatter.legend_elements()[0], 
                  labels=class_names, loc='upper right', fontsize=9)

    # --- 子图 1~3: 三类概率图 ---
    for i in range(3):
        ax = axes[idx, i + 1]
        # 渐变色：white → class color
        cmap_prob = mcolors.LinearSegmentedColormap.from_list(
            f'prob_{i}', ['white', colors[i]], N=256
        )
        contour = ax.contourf(xx, yy, probs[:, :, i], 
                              levels=np.linspace(0, 1, 11), 
                              cmap=cmap_prob, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_scatter, 
                   edgecolors='k', s=30, linewidth=0.5, alpha=0.8)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f'P({class_names[i]})', fontsize=11)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        fig.colorbar(contour, ax=ax, shrink=0.8, ticks=[0, 0.5, 1])

# ----------------------------
# 5. 保存 & 显示
# ----------------------------
plt.suptitle('Task 1: Comparison of Classifiers on Iris (2D, 3-class)', 
             fontsize=14, y=0.99)
plt.savefig('task1_classifiers_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('task1_classifiers_comparison.pdf', bbox_inches='tight')  # LaTeX 友好
plt.show()