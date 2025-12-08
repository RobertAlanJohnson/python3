import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
import numpy as np
import matplotlib
matplotlib.use('Agg') # 后台运行，适合服务器
import matplotlib.pyplot as plt
import cv2
import os

# ==========================================
# 1. 基础配置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'resnet18_stl10_lda.pth'  # 你的模型文件名

# STL-10 的目标类别
target_classes = [0, 2, 8]  # 飞机, 汽车, 船
class_names = ["Airplane", "Automobile", "Ship"]
label_map = {0: 0, 2: 1, 8: 2} # 映射

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

# ==========================================
# 2. Grad-CAM 核心算法
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        score = output[:, class_idx].squeeze()
        score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2)) 
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        # Resize 到 96x96 (STL-10 原生尺寸)
        cam = cv2.resize(cam, (96, 96)) 
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, output

# ==========================================
# 3. ⚠️ 关键：完美复刻模型结构
# ==========================================
print(f"📂 加载模型: {model_path} ...")

# 1. 初始化标准 ResNet18
model = resnet18(weights=None) 

# 2. ⚡️ 修改结构以匹配训练代码 (这就是解决报错的关键)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 3)

# 3. 加载参数
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ 模型参数加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()

model.to(device)
model.eval()

# 锁定 Layer4 最后一层
grad_cam = GradCAM(model, model.layer4[-1])

# ==========================================
# 4. 挑选图片并生成 Dashboard
# ==========================================
def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)

# 筛选置信度高的典型样本
indices = []
print("🔍 筛选最佳样本...")
# 简单的策略：每个类别遍历前50张，找分类正确的
for target_cls in target_classes:
    found = False
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == target_cls:
            # 简单验证一下模型预测是否正确，确保画出来的图是漂亮的
            with torch.no_grad():
                pred = model(img.unsqueeze(0).to(device)).argmax().item()
            if pred == label_map[target_cls]:
                indices.append(i)
                found = True
                break # 每个类只取一张
    if not found:
        print(f"⚠️ Warning: 没找到类别 {target_cls} 的合适样本")

# 设置绘图风格
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9), dpi=200)
fig.suptitle(f"STL-10 XAI Dashboard: ResNet18 Reasoning Process", fontsize=24, fontweight='bold', color='white', y=0.98)

print("🎨 正在渲染可视化...")

for i, idx in enumerate(indices):
    img_tensor, label = dataset[idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)
    mapped_label = label_map[label]
    
    # 获取热力图
    mask, output = grad_cam(input_tensor, mapped_label)
    
    # 准备图片
    raw_img = denormalize(img_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    cam_img = 0.5 * heatmap + 0.5 * raw_img
    cam_img = np.clip(cam_img, 0, 1)

    # --- 绘图 ---
    row = i * 5
    
    # 1. 原图
    ax1 = plt.subplot(3, 5, row + 1)
    ax1.imshow(raw_img)
    ax1.axis('off')
    ax1.text(-30, 48, class_names[mapped_label], fontsize=18, fontweight='bold', rotation=90, va='center', color='white')
    if i == 0: ax1.set_title("Input (96px)", fontsize=14, color='gray')

    # 2. 热力图
    ax2 = plt.subplot(3, 5, row + 2)
    ax2.imshow(mask, cmap='jet')
    ax2.axis('off')
    if i == 0: ax2.set_title("Attention", fontsize=14, color='gray')

    # 3. 叠加
    ax3 = plt.subplot(3, 5, row + 3)
    ax3.imshow(cam_img)
    ax3.axis('off')
    if i == 0: ax3.set_title("Reasoning", fontsize=14, color='gray')

    # 4. 置信度
    probs = F.softmax(output, dim=1).cpu().data.numpy()[0]
    ax4 = plt.subplot(3, 5, row + 4)
    colors = ['#FF4444', '#4444FF', '#44FF44'] # 红蓝绿
    bars = ax4.barh(class_names, probs, color=colors, alpha=0.8)
    ax4.set_xlim(0, 1.1)
    ax4.axis('off')
    # 标数值
    for bar in bars:
        w = bar.get_width()
        ax4.text(w + 0.05, bar.get_y()+0.4, f"{w:.1%}", color='white', fontsize=10)
    # 标类别名
    for j, name in enumerate(class_names):
        ax4.text(-0.1, j, name, ha='right', va='center', color='white', fontsize=10)
    if i == 0: ax4.set_title("Confidence", fontsize=14, color='gray')

    # 5. 特写 (智能裁剪)
    ax5 = plt.subplot(3, 5, row + 5)
    y_c, x_c = np.unravel_index(np.argmax(mask), mask.shape)
    margin = 24 # 裁剪范围
    y1, y2 = max(0, y_c-margin), min(96, y_c+margin)
    x1, x2 = max(0, x_c-margin), min(96, x_c+margin)
    crop = raw_img[y1:y2, x1:x2]
    
    # 调整 crop 大小一致显示
    if crop.size > 0:
        ax5.imshow(crop)
        # 画个框示意放大位置
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.5, edgecolor='yellow', facecolor='none')
        ax3.add_patch(rect)
        
    ax5.axis('off')
    for spine in ax5.spines.values():
        spine.set_edgecolor('yellow')
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    if i == 0: ax5.set_title("Focus Area", fontsize=14, color='gray')

plt.tight_layout(pad=1.5)
save_file = 'stl10_dashboard_fixed.png'
plt.savefig(save_file, dpi=200, bbox_inches='tight', facecolor='black')
print(f"✅ 可视化图已生成: {save_file}")