import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# ----------------------------
# 1. 设备与数据准备
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 常用均值与标准差，用于归一化
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

# 训练集增强：提升泛化能力（随机裁剪 + 水平翻转）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# 测试集仅做归一化，不增强
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# 加载完整 CIFAR-10 数据集
train_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_full = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 选取三个语义差异较大的类别：飞机(0)、汽车(1)、船(8)
target_classes = [0, 1, 8]
class_names = ["Airplane", "Automobile", "Ship"]

def get_subset(dataset, classes, limit=None):
    """
    从数据集中筛选指定类别样本。
    若指定 limit，则每个类别最多取 limit 个样本（用于控制测试集规模）。
    """
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    if limit:
        indices = indices[:limit]
    return Subset(dataset, indices)

# 构建子数据集与数据加载器
train_dataset = get_subset(train_full, target_classes)
test_dataset = get_subset(test_full, target_classes, limit=800)  # 控制测试点数量，兼顾可视化清晰度与计算效率

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# ----------------------------
# 2. 模型定义与训练
# ----------------------------

# 修改 ResNet18 以适配 CIFAR-10 的 32×32 输入尺寸：
#   - 首层卷积核改为 3×3（原为 7×7, stride=2），保留空间分辨率
#   - 移除 maxpool 层，防止特征图过早缩小
model = resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, len(target_classes))  # 输出维度匹配 3 类
model = model.to(device)

# 损失函数与优化器设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4  # L2 正则化，防止过拟合
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # 余弦退火学习率调度

# 标签映射：将原始标签 [0,1,8] 映射为连续索引 [0,1,2]，适配 CrossEntropyLoss
label_map = {0: 0, 1: 1, 8: 2}

# 训练循环（共 20 轮）
model.train()
for epoch in range(20):
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        # 转换标签至连续空间
        targets = torch.tensor([label_map[y.item()] for y in labels], device=device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

    scheduler.step()

    # 每 5 轮输出一次训练进度
    if (epoch + 1) % 5 == 0:
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:2d}/20 | Train Acc: {acc:5.2f}% | Avg Loss: {avg_loss:.4f}")

# 保存训练好的模型参数（仅 state_dict，便于后续加载）
torch.save(model.state_dict(), 'resnet18_cifar10_lda.pth')

# ----------------------------
# 3. 特征提取与降维（LDA）
# ----------------------------

# 构建特征提取器：移除最后的全连接分类层
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()

# 在测试集上提取 512 维全局平均池化后特征
features_list = []
labels_list = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        feats = feature_extractor(inputs).squeeze()  # shape: (B, 512)
        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())

# 合并所有批次特征与标签
X = np.concatenate(features_list, axis=0)          # shape: (N, 512)
y = np.concatenate(labels_list, axis=0)           # 原始标签
y_mapped = np.array([label_map[l] for l in y])    # 映射为 [0,1,2]

# 使用 LDA 将 512 维特征投影到 2 维（最大化类间可分性）
# 注意：LDA 是有监督方法，需提供标签 y_mapped 用于学习判别方向
lda = LDA(n_components=2)
X_2d = lda.fit_transform(X, y_mapped)  # shape: (N, 2)

# ----------------------------
# 4. 决策边界可视化
# ----------------------------

# 在 LDA 二维空间上训练 SVM（RBF 核），用于绘制平滑决策边界
# 选择较高 C（=10）增强对误分类的惩罚，使边界更贴合数据分布
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_2d, y_mapped)

# 创建网格用于绘制决策区域
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# 预测网格上每个点的类别
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = svm.predict(grid_points).reshape(xx.shape)

# 定义颜色映射：背景浅色（区分区域），前景深色（突出样本点）
cmap_light = ListedColormap(['#FFDDDD', '#DDDDFF', '#DDFFDD'])  # 背景：红/蓝/绿浅色
cmap_bold  = ListedColormap(['#CC0000', '#0000CC', '#008800'])  # 样本点：红/蓝/绿深色

# 绘图
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)  # 决策区域背景
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_mapped, cmap=cmap_bold,
            edgecolor='white', linewidth=0.5, s=60, alpha=0.9)  # 样本点

plt.title("2D Feature Projection via LDA with SVM Decision Boundaries", fontsize=14)
plt.xlabel("LDA Component 1 (Maximizes Between-Class Variance)")
plt.ylabel("LDA Component 2")

# 添加图例（避免默认色条，使用明确类别名）
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#CC0000', edgecolor='k', label='Airplane'),
    Patch(facecolor='#0000CC', edgecolor='k', label='Automobile'),
    Patch(facecolor='#008800', edgecolor='k', label='Ship')
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True)

plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('cifar10.png', dpi=300, bbox_inches='tight')
plt.show()