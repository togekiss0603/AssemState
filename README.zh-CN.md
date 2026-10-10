# AssemState

[论文 · arXiv:2610.08446](https://arxiv.org/abs/2610.08446) · [项目主页](https://assemstate.wirymouse1.chatgpt.site) · [完整英文说明](README.md)

对应论文 **AssemState: Manual and Physical-State-Guided Reasoning for Zero-shot Furniture Assembly**。

作者：Zhiyuan Qi、Jierui Li、Yifan Shen、Cheng Qian、Jiateng Liu（2026）。

公开版包含 v2m 装配树推理代码、维护版位姿 agent、论文原始提示词、逐操作实验记录和项目主页。主页展示正文与附录全部表格、7 张论文图，以及已确认的 5 张 Bench/applaro 输入样例图；完整数据集和家具网格不上传。

## 与论文附录的对应

装配树运行时的 M1–M4 提示词与论文原始文件逐字节一致；附录打印版只有明确记录的标点与符号替换。历史位姿提示词作为快照保留，与当前维护版 agent 的运行时提示词并不相同；完整历史 Judge 请求未找回，论文附录也说明了这一点。详见 [对应检查](docs/PAPER_ALIGNMENT.md)。

运行 `python scripts/audit_paper_alignment.py` 可检查提示词、网页表格覆盖和图片哈希；加上 `--paper-zip /path/to/manuscript.zip` 可直接对照论文源文件。引用格式见 [英文 README](README.md#citation)。

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
