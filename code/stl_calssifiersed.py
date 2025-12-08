import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器/后台运行
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
import matplotlib.patches as mpatches

# ==========================================
# 1. 配置与数据准备 (适配STL-10 + 增加样本量)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 正在使用设备: {device} (4090 全力输出)")

# STL-10 预处理（96x96原始尺寸，无需Resize）
transform_train = transforms.Compose([
    transforms.RandomCrop(96, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

print("⬇️ 加载 STL-10 数据...")
train_full = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
test_full = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)

# STL-10 类别映射：0=飞机, 2=汽车, 8=船
target_classes = [0, 2, 8]
class_names = ["Airplane", "Automobile", "Ship"]
label_map = {0: 0, 2: 1, 8: 2}

def get_subset(dataset, classes, limit=None):
    indices = [i for i, (_, y) in enumerate(dataset) if y in classes]
    if limit:
        indices = indices[:limit]
    return Subset(dataset, indices)

# 训练集全量，测试集取1000个样本（密集但不堆积）
train_dataset = get_subset(train_full, target_classes)
test_dataset = get_subset(test_full, target_classes, limit=1000)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# ==========================================
# 2. 训练模型 (20轮训练，确保极高分离度)
# ==========================================
print("🧠 定义 ResNet18 模型...")
model = resnet18(pretrained=True)
# STL-10是96x96，调整ResNet适配小尺寸
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 3)  # 3分类
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

print("🔥 开始 20 轮极速训练 (目标: 极高分离度)...")
epochs = 30
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        targets = torch.tensor([label_map[y.item()] for y in labels], device=device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    scheduler.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Acc: {100.*correct/total:.2f}% (Loss: {running_loss/len(train_loader):.4f})")

# 保存模型
save_path = 'resnet18_stl10_lda.pth'
torch.save(model.state_dict(), save_path)
print(f"✅ 训练完成！模型已保存到 {save_path}")

# ==========================================
# 3. 核心：LDA降维 (有监督，强制类别分离)
# ==========================================
print("📉 提取特征并使用 LDA 进行完美降维...")
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()

features_list = []
labels_list = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        feats = feature_extractor(inputs).squeeze().cpu().numpy()
        # 处理批量为1的情况
        if len(feats.shape) == 1:
            feats = feats[np.newaxis, :]
        features_list.append(feats)
        labels_list.append(labels.numpy())

X = np.concatenate(features_list)
y = np.concatenate(labels_list)
y_mapped = np.array([label_map[l] for l in y])

# LDA降维到2D（有监督，主动最大化类别间距离）
lda = LDA(n_components=2)
X_2d = lda.fit_transform(X, y_mapped)

# SVM拟合LDA结果，绘制丝滑边界
clf = SVC(kernel="rbf", C=10, gamma='scale')
clf.fit(X_2d, y_mapped)

# ==========================================
# 4. 高对比度可视化 (参考CIFAR-10逻辑)
# ==========================================
plt.figure(figsize=(11, 9), dpi=200)

# 超细密网格，边界更丝滑
x_min, x_max = X_2d[:, 0].min() - 2, X_2d[:, 0].max() + 2
y_min, y_max = X_2d[:, 1].min() - 2, X_2d[:, 1].max() + 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 高饱和度对比色（参考CIFAR-10，适配STL-10）
cmap_light = ListedColormap(['#FFDDDD', '#DDDDFF', '#DDFFDD'])  # 浅红/浅蓝/浅绿
cmap_bold = ListedColormap(['#CC0000', '#0000CC', '#008800'])   # 深红/深蓝/深绿

# 绘制背景（边界平滑）
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

# 绘制散点（白边+适中尺寸，不堆积）
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_mapped, cmap=cmap_bold,
            edgecolor='white', linewidth=0.8, s=60, alpha=0.9)

# 标题和标签（参考原样式）
plt.title("Perfect Classification Boundaries via ResNet18 + LDA\n(STL-10, Maximizing Class Separation)", 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("LDA Component 1 (Most Discriminative Axis)", fontsize=12, labelpad=10)
plt.ylabel("LDA Component 2 (Second Discriminative Axis)", fontsize=12, labelpad=10)

# 自定义图例（更醒目）
patches = [
    mpatches.Patch(color='#CC0000', label='Airplane'),
    mpatches.Patch(color='#0000CC', label='Automobile'),
    mpatches.Patch(color='#008800', label='Ship')
]
plt.legend(handles=patches, loc="upper right", title="Classes", 
           fontsize=12, framealpha=0.9, shadow=True)

# 美化网格
plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
plt.tight_layout()

# 保存高分辨率图
save_path = 'stl10_perfect_boundary.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 完美分类图已保存: {save_path}")