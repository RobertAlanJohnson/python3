import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

# ================= 1. Data and Model (3D Probability Map) =================
iris = load_iris()
mask = (iris.target == 1) | (iris.target == 2)
# Features: [Sepal Width (Color), Petal Length (X), Petal Width (Y)]
X_all = iris.data[mask][:, [1, 2, 3]] 
y = (iris.target[mask] == 2).astype(int)

feat_color = X_all[:, 0]
feat_x = X_all[:, 1]
feat_y = X_all[:, 2]

# Train SVM (parameters for steep decision boundary)
model = SVC(kernel='rbf', C=100, gamma=1.5)
model.fit(X_all, y)

# ================= 2. Generate Grid for 3D Probability Map =================
RESOLUTION = 120
pad = 1.5 # Wide field of view for Probability Map
x_min, x_max = feat_x.min() - pad, feat_x.max() + pad
y_min, y_max = feat_y.min() - pad, feat_y.max() + pad

xx, yy = np.meshgrid(np.linspace(x_min, x_max, RESOLUTION),
                     np.linspace(y_min, y_max, RESOLUTION))

# Baseline terrain for Probability Map
zz_fixed = np.full_like(xx, feat_color.mean())
Z_raw = model.decision_function(np.c_[zz_fixed.ravel(), xx.ravel(), yy.ravel()]).reshape(xx.shape)
max_abs = np.max(np.abs(Z_raw))
Z = np.clip((Z_raw / max_abs) * 100, -100, 100)

# ================= 3. Plotting (High-Contrast 3D Probability Map) =================
fig = plt.figure(figsize=(15, 12), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# --- A. High-contrast Z=0 Plane (Baseline of Probability Map) ---
z_zero = np.zeros_like(xx)

# 1. Solid color plane: DeepSkyBlue with high opacity
ax.plot_surface(xx, yy, z_zero, 
                color='deepskyblue',    
                alpha=0.4,              
                shade=False, zorder=1)

# 2. Plane grid: White grid on blue baseline
ax.plot_wireframe(xx, yy, z_zero, rstride=10, cstride=10, 
                  color='white', linewidth=0.5, alpha=0.6, zorder=2)

# 3. Decision boundary: Bold black line (core of Probability Map)
ax.contour(xx, yy, Z, zdir='z', offset=0, levels=[0], 
           colors='black', linewidths=4, zorder=10)

# --- B. Floating Probability Surface (Dark skeleton) ---
# Transparent surface
ax.plot_surface(xx, yy, Z, color='white', alpha=0.05, shade=False, zorder=5)
# Dark blue skeleton
ax.plot_wireframe(xx, yy, Z, rstride=6, cstride=6, 
                  color='midnightblue', linewidth=0.6, alpha=0.4, zorder=5)

# --- C. Wall Projections (Faded to highlight main plane) ---
proj_cmap = cm.coolwarm
ax.contourf(xx, yy, Z, zdir='z', offset=-100, cmap=proj_cmap, alpha=0.4) # Bottom
ax.contourf(xx, yy, Z, zdir='x', offset=x_min, cmap=proj_cmap, alpha=0.4) # Left
ax.contourf(xx, yy, Z, zdir='y', offset=y_max, cmap=proj_cmap, alpha=0.4) # Rear

# --- D. Data Points (Warm color for contrast) ---
z_pts = np.zeros_like(feat_x)
# Autumn_r (red->yellow) for high visibility on blue background
point_cmap = plt.cm.autumn_r 

# Class 0 (Low Probability)
sc1 = ax.scatter(feat_x[y==0], feat_y[y==0], z_pts[y==0],
                 c=feat_color[y==0], cmap=point_cmap, vmin=feat_color.min(), vmax=feat_color.max(),
                 s=100, marker='o', edgecolors='black', linewidth=1.2, 
                 label='Versicolor (Low Probability)', zorder=20)
# Class 1 (High Probability)
sc2 = ax.scatter(feat_x[y==1], feat_y[y==1], z_pts[y==1],
                 c=feat_color[y==1], cmap=point_cmap, vmin=feat_color.min(), vmax=feat_color.max(),
                 s=100, marker='^', edgecolors='black', linewidth=1.2, 
                 label='Virginica (High Probability)', zorder=20)

# ================= 4. Settings and Beautification =================
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min+0.5, y_max)
ax.set_zlim(-100, 100)

ax.set_xlabel('Petal Length (X)', fontsize=13, fontweight='bold', labelpad=15)
ax.set_ylabel('Petal Width (Y)', fontsize=13, fontweight='bold', labelpad=15)
ax.set_zlabel('Probability  Score (Z)', fontsize=13, fontweight='bold', labelpad=15)

ax.set_title("3D Probability  Map\n"
             " (Feature 3)", 
             fontsize=16, fontweight='bold', y=0.95)

# View angle for optimal 3D effect
ax.view_init(elev=35, azim=-60)

# Auxiliary grid (Faded)
ax.grid(True, linestyle=':', alpha=0.3, color='gray')
ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False

# Colorbar & Legend
cbar = plt.colorbar(sc1, ax=ax, shrink=0.6, pad=0.08)
cbar.set_label('Feature 3: Sepal Width (cm)', rotation=270, labelpad=20, fontweight='bold')
ax.legend(loc='upper left', title="Probability Class", frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig('3D_Probability_Map.png', dpi=300, bbox_inches='tight')
plt.show()