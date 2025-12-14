import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.patches as mpatches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_full = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)

target_classes = [0, 2, 8]
label_map = {0: 0, 2: 1, 8: 2}

def get_subset(dataset, classes, limit=None):
    indices = [i for i, (_, y) in enumerate(dataset) if y in classes]
    if limit:
        indices = indices[:limit]
    return Subset(dataset, indices)

test_dataset = get_subset(test_full, target_classes, limit=1000)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

class DualBranchResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.branch_a_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.branch_a_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.branch_b_upsample = nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)
        self.branch_b_conv = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        feat_a = self.layer2(x)
        feat_b = self.layer3(feat_a)  # layer3 接收 layer2 输出
        
        feat_a = self.branch_a_pool(feat_a)
        feat_a = self.branch_a_conv(feat_a)
        feat_b = self.branch_b_upsample(feat_b)
        feat_b = self.branch_b_conv(feat_b)
        
        combined = torch.cat([feat_a, feat_b], dim=1)  # 16×16×2
        return combined.view(combined.size(0), -1)     # (B, 512)

feature_extractor = DualBranchResNet().to(device)
feature_extractor.eval()

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

lda = LinearDiscriminantAnalysis(n_components=2)
X_2d = lda.fit_transform(X, y_mapped)

clf = SVC(kernel="rbf", C=10, gamma='scale')
clf.fit(X_2d, y_mapped)

x_min, x_max = X_2d[:, 0].min() - 2, X_2d[:, 0].max() + 2
y_min, y_max = X_2d[:, 1].min() - 2, X_2d[:, 1].max() + 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(11, 9), dpi=200)

cmap_light = ListedColormap(['#FFDDDD', '#DDDDFF', '#DDFFDD'])
cmap_bold = ListedColormap(['#CC0000', '#0000CC', '#008800'])

plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_mapped, cmap=cmap_bold,
            edgecolor='white', linewidth=0.8, s=60, alpha=0.9)

plt.title("ResNet Dual-Branch Spatial Features + LDA + SVM\n(STL-10, 16×16×2 → 512-dim)",
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("LDA Component 1", fontsize=12, labelpad=10)
plt.ylabel("LDA Component 2", fontsize=12, labelpad=10)

patches = [
    mpatches.Patch(color='#CC0000', label='Airplane'),
    mpatches.Patch(color='#0000CC', label='Automobile'),
    mpatches.Patch(color='#008800', label='Ship')
]
plt.legend(handles=patches, loc="upper right", title="Classes", fontsize=12, framealpha=0.9, shadow=True)
plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
plt.tight_layout()

save_path = 'stl10.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 分类图已保存: {save_path}")