# AssemState

[项目主页](https://assemstate.wirymouse1.chatgpt.site) · [完整英文说明](README.md)

对应论文 **AssemState: Manual and Physical-State-Guided Reasoning for Zero-shot Furniture Assembly**。

公开版包含 v2m 装配树推理代码、维护版位姿 agent、论文原始提示词、逐操作实验记录和项目主页。主页仅展示已确认的 5 张 Bench/applaro 输入样例图；完整数据集和家具网格不上传。

## Clone 后直接复核论文表格

```bash
git clone https://github.com/togekiss0603/AssemState.git
cd AssemState
python scripts/reproduce_tables.py
```

不需要密钥、GPU、数据网格或额外 Python 依赖。脚本从 7 组、共 1,701 条 pose/release 记录重算表格，保存到 `outputs/tables/`。这是历史结果复核，不是重新调用模型或运行仿真。

## 环境和真实推理

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

依赖未分发样例资产的测试会明确跳过。自行准备数据并配置 `.env` 后：

```bash
source .env
python scripts/run_tree.py --data-root /path/to/assemblydata --names Bench/applaro
```

`run_tree.py` 固定调用 `run_anchor.py` 的 **v2m**，实际规划提示词为 `planning_anchor_v2m.txt`，与论文压缩包逐字一致。

数据目录、位姿命令、密钥配置和输出说明见 [README](README.md) 与 [数据文档](docs/DATA.md)。五张网页展示图片并不构成完整推理输入，不能代替手册全页、网格和相关标注。

## 复现范围

论文装配树评测依赖预先提供的编号步骤 mask；位姿评测依赖正确操作和前序状态，预测先评分，再用 GT 构造下一步上下文。二者分开评测，不是预测树驱动的整件家具端到端成功率。

缺失项不全是消融：主方法 v2m 代码在，但完整历史预测未恢复；三轮位姿自检是主实验，其逐操作记录在，但原始 Judge 等运行脚本不完整。维护版 agent 不能冒充历史原版。具体见 [评测协议](docs/PROTOCOL.md)。
