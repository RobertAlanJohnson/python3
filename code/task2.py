import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# ----------------------------
# 1. 数据加载与预处理
# ----------------------------
iris = load_iris()
mask = (iris.target == 1) | (iris.target == 2)
X = iris.data[mask][:, [2, 3, 0]]  # PetalL(x), PetalW(y), SepalL(z)
y = (iris.target[mask] == 2).astype(int)

print("特征顺序：x=Petal Length, y=Petal Width, z=Sepal Length")
print("X shape:", X.shape)

# ----------------------------
# 2. 训练逻辑回归模型
# ----------------------------
model = LogisticRegression()
model.fit(X, y)
w = model.coef_[0]
b = model.intercept_[0]
print("权重 w =", w)
print("偏置 b =", b)

# ----------------------------
# 3. 构建决策面网格
# ----------------------------
x = np.linspace(X[:,0].min() - 0.2, X[:,0].max() + 0.2, 30)
y_grid = np.linspace(X[:,1].min() - 0.2, X[:,1].max() + 0.2, 30)
X_grid, Y_grid = np.meshgrid(x, y_grid)

if abs(w[2]) < 1e-4:
    print("⚠️ w_z 接近 0，改用 x-z 网格解 y")
    x = np.linspace(X[:,0].min() - 0.2, X[:,0].max() + 0.2, 30)
    z = np.linspace(X[:,2].min() - 0.2, X[:,2].max() + 0.2, 30)
    X_grid, Z_grid = np.meshgrid(x, z)
    if abs(w[1]) < 1e-4:
        print("⚠️ w_y 也接近 0，改用 y-z 网格解 x")
        y_grid = np.linspace(X[:,1].min() - 0.2, X[:,1].max() + 0.2, 30)
        z = np.linspace(X[:,2].min() - 0.2, X[:,2].max() + 0.2, 30)
        Y_grid, Z_grid = np.meshgrid(y_grid, z)
        X_surf = -(w[1]*Y_grid + w[2]*Z_grid + b) / w[0]
        Y_surf, Z_surf = Y_grid, Z_grid
    else:
        Y_surf = -(w[0]*X_grid + w[2]*Z_grid + b) / w[1]
        X_surf, Z_surf = X_grid, Z_grid
else:
    Z_surf = -(w[0]*X_grid + w[1]*Y_grid + b) / w[2]
    X_surf, Y_surf = X_grid, Y_grid

# ----------------------------
# 4. 绘图（彻底解决左侧遮挡）
# ----------------------------
# 1. 进一步放大画布，重点预留左侧空间
fig = plt.figure(figsize=(12, 9))  
ax = fig.add_subplot(111, projection='3d')

# 绘制决策面
ax.plot_surface(X_surf, Y_surf, Z_surf, color='gray', alpha=0.5, edgecolor='none')

# 绘制数据点
ax.scatter(X[y==0, 0], X[y==0, 1], X[y==0, 2], 
           c='green', label='Versicolor', s=60, edgecolors='k')
ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], 
           c='red',   label='Virginica',  s=60, edgecolors='k')

# 2. 核心：大幅增加Y轴标签的左侧偏移 + 放大字体
ax.set_xlabel('Petal Length (cm)', fontsize=13, labelpad=15)  # X轴标签间距
ax.set_ylabel('Petal Width (cm)', fontsize=13, labelpad=25)  # Y轴（左侧）标签大幅右移（pad=25）
ax.set_zlabel('Sepal Length (cm)', fontsize=13, labelpad=15)  # Z轴标签间距

# 3. 调整视角：让左侧完全露出来（azim增大，elev略降）
ax.view_init(elev=20, azim=55)  # 方位角azim从45→55，彻底避开左侧遮挡

# 4. 手动调整画布边距：左侧留足空间（left=0.2）
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)  

# 5. 图例移到右侧，避免挤占左侧空间
ax.legend(loc='upper right', fontsize=11)
ax.set_title('Task 2: 3D Decision Boundary (Stable Features)', fontsize=15, pad=20)

# 6. 保存时增加左侧pad，防止裁剪
plt.savefig('task2_stable.png', dpi=300, bbox_inches='tight', pad_inches=1.0)
plt.show()