import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像归一化参数（CIFAR-10）
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

# 测试集数据变换（无增强）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# 加载 CIFAR-10 测试集
test_full = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 选定类别：飞机(0)、汽车(1)、船(8)
target_classes = [0, 1, 8]
label_map = {0: 0, 1: 1, 8: 2}

def get_subset(dataset, classes, limit=None):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    if limit:
        indices = indices[:limit]
    return Subset(dataset, indices)

# 构建子集（限制样本数以提升可视化清晰度）
test_dataset = get_subset(test_full, target_classes, limit=800)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# 双分支 ResNet 特征提取器：输出 16×16×2 空间特征图 → 展平为 512 维
class DualBranchResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        # 替换首层以适配 32×32 输入
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1  # 32×32×64
        self.layer2 = resnet.layer2  # 16×16×128
        self.layer3 = resnet.layer3  # 8×8×256

        # 分支 A：layer2 → 1×1 卷积 → 1 通道
        self.branch_a_conv = nn.Conv2d(128, 1, kernel_size=1)
        # 分支 B：layer3 → 上采样 → 1×1 卷积 → 1 通道
        self.branch_b_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.branch_b_conv = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        feat_a = self.layer2(x)          # 16×16×128
        feat_b = self.layer3(feat_a)     # 8×8×256
        
        feat_a = self.branch_a_conv(feat_a)              # 16×16×1
        feat_b = self.branch_b_upsample(feat_b)          # 16×16×256
        feat_b = self.branch_b_conv(feat_b)              # 16×16×1
        
        combined = torch.cat([feat_a, feat_b], dim=1)    # 16×16×2
        return combined.view(combined.size(0), -1)       # (B, 512)

# 实例化并切换至评估模式
feature_extractor = DualBranchResNet().to(device)
feature_extractor.eval()

# 特征提取
features_list = []
labels_list = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        feats = feature_extractor(inputs)
        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())

X = np.concatenate(features_list, axis=0)
y = np.concatenate(labels_list, axis=0)
y_mapped = np.array([label_map[label] for label in y])

# LDA 降维至 2 维
lda = LinearDiscriminantAnalysis(n_components=2)
X_2d = lda.fit_transform(X, y_mapped)

# 在 2D 空间训练 RBF-SVM
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_2d, y_mapped)

# 创建网格用于绘制决策边界
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = svm.predict(grid).reshape(xx.shape)

# 绘图配置
cmap_light = ListedColormap(['#FFDDDD', '#DDDDFF', '#DDFFDD'])
cmap_bold  = ListedColormap(['#CC0000', '#0000CC', '#008800'])

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_mapped, cmap=cmap_bold,
            edgecolor='white', linewidth=0.5, s=60, alpha=0.9)

plt.title("LDA + RBF-SVM on Dual-Branch Spatial Features (16×16×2 → 512-dim)")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#CC0000', label='Airplane'),
    Patch(facecolor='#0000CC', label='Automobile'),
    Patch(facecolor='#008800', label='Ship')
]
plt.legend(handles=legend_elements, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('cifar10.png', dpi=300, bbox_inches='tight')
plt.show()