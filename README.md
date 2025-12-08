
## 🎯 项目目标

1.  **基础建模**：使用 `scikit-learn` 对Iris数据集进行探索性分析，并在二维/三维空间中可视化不同分类器（Logistic Regression, SVM, Decision Tree）的决策边界。
2.  **创新拓展**：选用高难度的图像数据集（STL-10, CIFAR-10），构建“**ResNet18特征提取 → LDA降维 → SVM分类 → Grad-CAM可解释性**”的技术闭环，实现模型决策过程的可视化。

## 🛠️ 核心技术栈

*   **Python**: 主要编程语言。
*   **scikit-learn**: 用于传统机器学习模型训练与评估。
*   **PyTorch**: 用于加载预训练的ResNet18模型进行深度特征提取。
*   **matplotlib / seaborn**: 用于数据可视化。
*   **OpenCV / PIL**: 用于图像处理和Grad-CAM热力图生成。
*   **Jupyter Notebook** (可选): 用于交互式开发和结果展示。

## 📊 主要成果

### 1. Iris 数据集可视化

我们成功绘制了多种分类器在二维特征空间中的决策边界，并实现了三维空间下的决策曲面和概率体渲染。

#### 二维决策边界示例
![Iris 2D Decision Boundary](task1.png)

#### 三维概率体渲染示例
![Iris 3D Probability Map](3D_Probability_Map.png)

### 2. 图像数据集分类与可解释性 (XAI)

我们在STL-10和CIFAR-10数据集上取得了优异的成绩，并构建了直观的XAI Dashboard，揭示了模型的决策依据。

#### STL-10 分类结果示例
![STL-10 Dataset Sample](stl10.png)

#### STL-10 XAI Dashboard 示例
![STL-10 XAI Dashboard](stl10_dashboard.png)
*Dashboard展示了原图、Grad-CAM热力图（红色区域为模型关注点）和分类置信度。*

#### CIFAR-10 分类结果示例
![CIFAR-10 Dataset Sample](cifar10.png)

#### CIFAR-10 XAI Dashboard 示例
![CIFAR-10 XAI Dashboard](cifar10_dashboard.png)
*同样展示了模型如何“看”待图像并做出决策。*

## 🚀 如何运行

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/RobertAlanJohnson/python2.git
    cd python2
    ```

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt  # 如果存在此文件，否则请手动安装所需库
    ```
    常用依赖包包括: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `torchvision`, `opencv-python`。

3.  **运行脚本**:
    *   运行Iris相关任务:
        ```bash
        python task1.py
        python task2.py
        python task3.py
        ```
    *   运行图像分类与可视化任务:
        ```bash
        python stl_calssifiersed.py
        python cifar10.py
        ```

## 📝 说明

*   本项目代码已按实验报告要求完成所有指定任务。
*   所有生成的图片（`.png`）均已保存在仓库中，便于直接查看结果。
*   `dashboard.py` 是一个通用的XAI可视化模块，可供其他项目复用。
*   在STL-10数据集上实现了 **98.6%** 的测试准确率。

## 📬 联系方式

如有任何问题或建议，欢迎提交Issue。

---
> **注**: 本项目为课程作业，旨在学习和实践。代码和结果仅供学术交流之用。