import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'resnet18_stl10_lda.pth'

target_classes = [0, 2, 8]
class_names = ["Airplane", "Automobile", "Ship"]
label_map = {0: 0, 2: 1, 8: 2}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

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
        cam = cv2.resize(cam, (96, 96))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, output

class DualBranchResNetForGradCAM(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = resnet18(weights=None)
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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        feat_a = self.layer2(x)
        feat_b = self.layer3(feat_a)  # fixed: layer3 takes feat_a, not x
        
        feat_a = self.branch_a_pool(feat_a)
        feat_a = self.branch_a_conv(feat_a)
        feat_b = self.branch_b_upsample(feat_b)
        feat_b = self.branch_b_conv(feat_b)
        
        combined = torch.cat([feat_a, feat_b], dim=1)  # 16×16×2
        return self.classifier(combined)

print(f"📂 加载模型: {model_path} ...")
model = DualBranchResNetForGradCAM(num_classes=3)

try:
    # Use weights_only=True for security if PyTorch ≥2.4
    load_kwargs = {"weights_only": True} if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals") else {}
    state_dict = torch.load(model_path, map_location=device, **load_kwargs)
    model_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    print("✅ 模型参数部分加载成功！")
except Exception as e:
    print(f"⚠️ 模型加载失败，使用随机初始化权重: {e}")

model.to(device)
model.eval()

grad_cam = GradCAM(model, model.layer2[-1])

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)

indices = []
print("🔍 筛选最佳样本...")
for target_cls in target_classes:
    found = False
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == target_cls:
            with torch.no_grad():
                pred = model(img.unsqueeze(0).to(device)).argmax().item()
            if pred == label_map[target_cls]:
                indices.append(i)
                found = True
                break
    if not found:
        print(f"⚠️ 未找到类别 {target_cls} 的合适样本")

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9), dpi=200)
fig.suptitle("STL-10 XAI Dashboard: ResNet18 Reasoning Process", fontsize=24, fontweight='bold', color='white', y=0.98)

print("🎨 正在渲染可视化...")

for i, idx in enumerate(indices):
    img_tensor, label = dataset[idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)
    mapped_label = label_map[label]
    
    mask, output = grad_cam(input_tensor, mapped_label)
    
    raw_img = denormalize(img_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    cam_img = 0.5 * heatmap + 0.5 * raw_img
    cam_img = np.clip(cam_img, 0, 1)

    row = i * 5
    
    ax1 = plt.subplot(3, 5, row + 1)
    ax1.imshow(raw_img)
    ax1.axis('off')
    ax1.text(-30, 48, class_names[mapped_label], fontsize=18, fontweight='bold', rotation=90, va='center', color='white')
    if i == 0: ax1.set_title("Input (96px)", fontsize=14, color='gray')

    ax2 = plt.subplot(3, 5, row + 2)
    ax2.imshow(mask, cmap='jet')
    ax2.axis('off')
    if i == 0: ax2.set_title("Attention", fontsize=14, color='gray')

    ax3 = plt.subplot(3, 5, row + 3)
    ax3.imshow(cam_img)
    ax3.axis('off')
    if i == 0: ax3.set_title("Reasoning", fontsize=14, color='gray')

    probs = F.softmax(output, dim=1).cpu().data.numpy()[0]
    ax4 = plt.subplot(3, 5, row + 4)
    colors = ['#FF4444', '#4444FF', '#44FF44']
    bars = ax4.barh(class_names, probs, color=colors, alpha=0.8)
    ax4.set_xlim(0, 1.1)
    ax4.axis('off')
    for bar in bars:
        w = bar.get_width()
        ax4.text(w + 0.05, bar.get_y() + 0.4, f"{w:.1%}", color='white', fontsize=10)
    for j, name in enumerate(class_names):
        ax4.text(-0.1, j, name, ha='right', va='center', color='white', fontsize=10)
    if i == 0: ax4.set_title("Confidence", fontsize=14, color='gray')

    ax5 = plt.subplot(3, 5, row + 5)
    y_c, x_c = np.unravel_index(np.argmax(mask), mask.shape)
    margin = 24
    y1, y2 = max(0, y_c - margin), min(96, y_c + margin)
    x1, x2 = max(0, x_c - margin), min(96, x_c + margin)
    crop = raw_img[y1:y2, x1:x2]
    
    if crop.size > 0:
        ax5.imshow(crop)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='yellow', facecolor='none')
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