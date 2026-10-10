<h2 align="center">
  <a href="https://owensun2004.github.io/Furniture-Assembly-Web/">Manual2Skill: Learning to Read Manuals and Acquire Robotic Skills for Furniture Assembly Using Vision-Language Models</a>
</h2>

<h5 align="center">

[Chenrui Tie](https://crtie.github.io/),
[Shengxiang Sun](https://owensun2004.github.io/),
[Jinxuan Zhu](https://www.linkedin.com/in/jinxuan-zhu-08a8972b7/),
[Yiwei Liu](https://lew1sin.github.io/),
[Jingxiang Guo](https://borisguo6.github.io/),
[Yue Hu](https://blog.csdn.net/xyyxyyx),
[Haonan Chen](https://github.com/chenhn02),
[Junting Chen](https://sgtvincent.github.io/),
[Ruihai Wu](https://warshallrho.github.io/),
[Lin Shao](https://linsats.github.io/)

<font color="black"><strong>RSS 2025</strong></font>

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2502.10090)
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://owensun2004.github.io/Furniture-Assembly-Web)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15XH8jhqodE4-GuiIe_IRrwjaeZt6uiQj?usp=sharing)

</h5>

![Demo Preview](assets/fig1.png)

---

## 简介 | Overview

**Manual2Skill** 是一个让机器人能够"读懂"家具组装手册并执行装配任务的框架。系统以 IKEA 风格的手册图片和零件预摆放场景为输入，自动生成可执行的层级化装配计划，并在物理仿真中完成验证。

**Manual2Skill** enables robots to perform complex furniture assembly tasks by reading IKEA-style manuals. Given a manual and a pre-assembly scene, it generates a hierarchical assembly plan and validates it end-to-end in physics simulation.

### 系统流程 | Pipeline

```
手册 (PDF) ──► [VLM 规划生成] ──► [姿态估计] ──► [物理仿真验证]
Manual        Assembly Graph      6D Poses      PyBullet Sim
```

系统由三个核心模块组成：

1. **层级化装配图生成 (VLM Assembly Plan Generation)**  
   使用视觉语言模型（GPT-4o / o1 / o3）解析手册，输出嵌套列表形式的装配树 `tree.json`。  
   示例：`[[[4, 0, 1], 2], 3]` 表示先组装零件 4+0+1，再连接零件 2，最后连接零件 3。

2. **逐步装配姿态估计 (Per-step Pose Estimation)**  
   图神经网络（DeepLab V3+ 图像编码 + PointNet2 点云编码）预测每个零件的 6D 姿态（位置 + 四元数）。

   另外，本仓库新增了一个基于多视图 VLM 的装配位姿估计 pipeline，可直接读取
   `/home/data/assemblydata` 下的手册页、OBJ 零件、五视图预摆放场景、投影 pkl 和
   `main_data.json` 装配树，输出每个零件从初始场景移动到最终成品状态的 `T` / `R`
   矩阵和 waypoint 路径。详细说明见
   [`vlm_pose_estimation/README.md`](vlm_pose_estimation/README.md)。

3. **物理仿真与验证 (Physics-based Simulation)**  
   基于 PyBullet，采用 LERP/SLERP 运动规划逐步执行装配，检测碰撞、位姿误差和接触成功率。

### 数据集 | Dataset

102 件 IKEA 家具，涵盖 6 类：**椅子 Chair**（59）、**桌子 Table**（21）、**长凳 Bench**（12）、**书桌 Desk**（6）、**置物架 Shelf**（5）、**其他 Misc**（13）。

---

## 环境配置 | Setup

### 前置要求 | Prerequisites

- Python 3.10
- CUDA 11.7
- Blender 3.6.19（场景渲染）
- OpenAI API 访问权限

### 安装步骤 | Installation

**1. 克隆仓库**

```bash
git clone https://github.com/owensun2004/Manual2Skill.git
cd Manual2Skill
```

**2. 创建并激活 conda 环境**

```bash
conda env create -f environment.yml
conda activate manual
```

主要依赖包：

| 类别 | 包 |
|------|-----|
| 深度学习 | PyTorch 2.0.1, PyTorch3D 0.7.4, TorchVision 0.15.2 |
| 图神经网络 | PyTorch Geometric 2.7.0, NetworkX 3.4.2, igraph 1.0.0 |
| 计算机视觉 | OpenCV 4.13.0, Pillow 11.1.0, ImageIO 2.37.3 |
| 3D 处理 | Trimesh 4.11.5, PyBullet 3.2.7 |
| VLM 接口 | OpenAI 2.8.0, httpx 0.28.1 |
| 科学计算 | NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.7.2 |
| 可视化 | Matplotlib 3.10.8, Plotly 6.7.0 |

**3. 安装 Blender**

```bash
wget https://download.blender.org/release/Blender3.6/blender-3.6.19-linux-x64.tar.xz
tar -xf blender-3.6.19-linux-x64.tar.xz
cd blender-3.6.19-linux-x64
sudo ln -s $(pwd)/blender /usr/local/bin/blender
cd ..
rm blender-3.6.19-linux-x64.tar.xz
```

**4. （可选）PyTorch3D 预编译安装**

若 conda 安装 PyTorch3D 失败，使用预编译包：

```bash
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html
```

参考：[PyTorch3D Installation Guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

### API 配置 | API Configuration

**直连 OpenAI：**
```bash
export OPENAI_API_KEY="your-api-key"
```

**中转端点（OpenAI 兼容接口）：**
```bash
export OPENAI_API_KEY="your-relay-token"
export OPENAI_BASE_URL="https://api.bltcy.ai/v1"
export OPENAI_MODEL="mimo-v2.5-pro"        # 全局模型（可选）
export STAGE1_MODEL="mimo-v2.5-pro"        # Stage 1 专用模型（可选）
export STAGE2_MODEL="mimo-v2.5-pro"        # Stage 2 专用模型（可选）
export CONVERT_MODEL="mimo-v2.5-pro"       # 转换阶段模型（可选）
export REQUEST_TIMEOUT="60"
```

模型优先级：`STAGE{N}_MODEL` → `OPENAI_MODEL` → `--model` 命令行参数

测试 API 连通性：
```bash
cd VLM_assembly_plan_gen
python inference/test_api.py
```

---

## 使用方法 | Usage

### 模块一：层级化装配图生成

#### 数据下载

```bash
cd VLM_assembly_plan_gen
mkdir data
gdown https://drive.google.com/uc?id=1hPesH_zd_NMd842JGaXaUxkviLU2Th4L
unzip data.zip -d ./data
```

数据目录结构：
```
data/
├── parts/                  # OBJ 格式零件 3D 网格
├── pdfs/                   # 手册页（PNG）
├── mask/                   # 零件分割掩码
└── preassembly_scenes/     # 标注好零件编号的预摆放场景图
```

#### 预摆放场景生成（Blender）

```bash
python scene_gen/generator.py --rand_translate true --rand_rotate true
```

- `--rand_translate true`：随机打乱零件位置
- `--rand_rotate true`：随机旋转零件
- 不加参数则使用原始摆放

生成图片保存至 `data/preassembly_scenes/scene_rot.png`（含标注版 `scene_rot_annotated.png`）。

#### VLM 推理

```bash
cd VLM_assembly_plan_gen

# 快速测试（前 10 件家具）
python inference/run.py --start 0 --end 10 --model mimo-v2.5-pro --debug

# 完整推理（全部 102 件）
python inference/run.py
```

完整参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--start` | 起始索引 | 0 |
| `--end` | 结束索引 | 102 |
| `--model` | 使用的 VLM 模型 | mimo-v2.5-pro |
| `--temperature` | 生成随机性 (0-1) | 0.1 |
| `--prompt_type` | 手册图片格式 (`numbered` / `not_numbered`) | numbered |
| `--scene_type` | 场景类型 (`original` / `varied`) | original |
| `--debug` | 启用详细日志 | False |

输出：`outputs/[timestamp]/[category]/[name]/tree.json`

> 资源消耗参考：每件家具约 15 张图 + 1700 词，耗时 1–2 分钟。

三阶段推理流程：
- **Stage 1** (`stage1_associate.py`)：从场景图和手册页中识别所需零件
- **Stage 2** (`stage2_planning.py`)：生成逐步装配指令
- **Convert** (`convert.py`)：将装配指令转换为嵌套树结构 `tree.json`

#### 评估

```bash
cd eval
bash manual_generation/test.sh
```

---

### 模块二：逐步装配姿态估计

#### 数据下载

```bash
cd pose_estimation
mkdir data
gdown https://drive.google.com/uc?id=1hPesH_zd_NMd842JGaXaUxkviLU2Th4L
unzip data.zip -d ./data
```

期望目录结构：
```
pose_estimation/data/
└── partnet_chair_mono/
    └── {item_id}/
        └── {step_id}/
            ├── depth_image.png
            ├── rgb_image.png
            └── ...
```

#### 训练

配置文件：`pose_estimation/configs/base_pose_estimate.py`

```bash
cd pose_estimation/pose_estimation
python train_new_no_dist.py
```

模型权重保存至 `pose_estimation/logs/`。

> 首次运行会序列化数据（pickle），耗时较长，属正常现象。

#### 推理与评估

```bash
cd pose_estimation/pose_estimation
# 在 eval.py 中设置 cfg.train.pretrained_weights 为 checkpoint 路径
# 示例：./logs/GNNNetwork_partnet_chair_mono_04-19-21-08-36/best.ckpt
python eval.py
```

---

### 模块三：物理仿真验证

```bash
cd simulation

# 单件家具
python run_vlm_assembly.py --category Bench --name applaro

# 批量评估
python run_overall_eval.py
```

仿真输出至 `simulation/outputs/`。

**成功判定标准（三类失败）：**

| 失败类型 | 条件 |
|----------|------|
| `POSE_ERROR` | 最终位置误差 > 2cm 或旋转误差 > 0.1 rad |
| `TRAJECTORY_COLLISION` | 运动过程中与已装配部件发生碰撞 |
| `FLOATING_PART` | 零件与主体结构无接触 |

---

## 项目结构 | Project Structure

```
Manual2Skill/
├── VLM_assembly_plan_gen/          # 装配图生成
│   ├── data/                        # 手册与场景数据
│   ├── inference/                   # VLM 推理流水线
│   │   ├── run.py                   # 主入口
│   │   ├── stage1_associate.py      # 零件识别
│   │   ├── stage2_planning.py       # 装配规划
│   │   └── convert.py               # 树结构转换
│   ├── scene_gen/                   # Blender 场景生成
│   ├── prompts/                     # VLM 提示词
│   └── eval/                        # 评估脚本
├── pose_estimation/                 # 姿态估计
│   ├── pose_estimation/
│   │   ├── train_new_no_dist.py     # 训练脚本
│   │   └── eval.py                  # 评估脚本
│   └── configs/                     # 训练配置
├── simulation/                      # PyBullet 仿真
│   ├── pybullet_env.py              # 仿真环境
│   ├── planner.py                   # 运动规划（LERP/SLERP）
│   ├── graph_runner.py              # 装配树执行（后序 DFS）
│   ├── success_checker.py           # 成功检测
│   └── run_overall_eval.py          # 批量评估入口
├── environment.yml
└── README.md
```

---

## 引用 | Citation

如果本工作对您有帮助，请引用：

```bibtex
@inproceedings{tie2025manual,
  title     = {Manual2Skill: Learning to Read Manuals and Acquire Robotic Skills for Furniture Assembly Using Vision-Language Models},
  author    = {Tie, Chenrui and Sun, Shengxiang and Zhu, Jinxuan and Liu, Yiwei and Guo, Jingxiang and Hu, Yue and Chen, Haonan and Chen, Junting and Wu, Ruihai and Shao, Lin},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year      = {2025}
}
```

---

## 致谢 | Acknowledgements

部分代码参考自 [IKEA-Manual](https://cs.stanford.edu/~rcwang/projects/ikea_manual/)，感谢作者的出色工作！

---

## 常见问题 | Troubleshooting

**PyTorch3D 安装失败** → 使用预编译 wheel（见安装步骤 4）。

**API 超时** → 增大 `REQUEST_TIMEOUT` 环境变量。

**Blender 未找到** → 确认已将 Blender 正确软链接至 `/usr/local/bin/blender`。

**CUDA 显存不足** → 减小训练配置中的 batch size，或使用更大显存的 GPU。

其他问题请在 GitHub 提 Issue。

---

## License

See [LICENSE](LICENSE) for details.
