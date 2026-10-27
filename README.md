# PGPS Geometry OPSD

将已核对的 **Qwen3-VL-8B-Instruct、原六卡 Vision-OPD/verl 实现**迁移到独立 Linux 服务器。提供两个配方：原教师提示词 `original`，以及只给教师追加最终约束的 `hidden_guidance`。

代码、数据、环境、模型、实验产物各自独立。GitHub 仓库不包含数据集图片、模型权重、Python 环境、历史作答、实验日志或密钥。

## 固定的实验设置

| 项目 | 配置 |
|---|---|
| 模型 | 原始 Qwen3-VL-8B-Instruct；学生与冻结教师相同大小 |
| 训练数据 | PGPS9K 8021题、8022行；保留旧行序与原补齐行 |
| 学生输入 | 实际训练图片＋原题，无选项、解析或答案 |
| 教师输入 | 同图同题＋原始结构/语义解析；约束仅追加在教师侧 |
| 卡数 / batch / rollout | **6卡 / 12 / 1** |
| 输出上限 / 学习率 | 4096 tokens / 1e-6 |
| 采样 | temperature=1, top_p=1, top_k=-1, presence=0, repetition=1, frequency=0 |
| 训练时长 | 1 epoch，669次更新 |
| 保存 | 100、200、300、400、500、600、669步；TensorBoard和可选W&B |
| Attention / loss | 原FA2去padding、原六卡归一化；不含四卡适配或新版vLLM补丁 |

**8×H100 服务器仍默认用6张卡训练**，保持原 batch 与损失实现。其余2张卡可留给独立评测。八卡全部用于训练需要另外验证，当前入口会拒绝静默改成八个训练rank。

## 目录

```text
framework/          固定提交＋历史补丁的完整训练框架（482个原文件）
src/geo_opsd/       可迁移的配置、数据、启动、导出与检查入口
scripts/            手动执行的命令行入口
configs/            六卡配方、教师约束、模型训练元数据、local配置示例
environment/        Conda配置、版本锁、FA2校验清单、安装/验证脚本
provenance/         源码校验、配置基准和版本证据
tests/              CPU单元测试，不调用模型或API
docs/               数据迁移、训练和环境说明
```

## 新服务器快速开始

### 1. 取代码

```bash
git clone https://github.com/JunC3927/opsd-pgps.git
cd opsd-pgps
```

### 2. 安装独立Conda环境

已安装Conda的Linux x86_64服务器示例，路径自行修改：

```bash
bash environment/install.sh \
  --prefix /your/local-disk/envs/geometry-opsd \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --download-flash-attn
conda activate /your/local-disk/envs/geometry-opsd
```

原FA2 wheel也可以独立传输，用 `--flash-attn-wheel /path/to/wheel.whl` 替代下载参数。默认沿用系统代理；需要直连时显式加 `--no-proxy`。安装中不会使用GPU、启动训练或修改其他环境。

依赖固定为 torch 2.8.0、transformers 4.57.3、vLLM 0.11.0、FA2 2.8.3 等旧版本，**不要直接升级到最新版**。完整说明见 [环境说明](docs/ENVIRONMENT.md)。

### 3. 放置模型和独立数据包

- 在新服务器下载官方 `Qwen/Qwen3-VL-8B-Instruct`，或者传输自己已核验的原始权重；不要用训后权重初始化。
- 把私下传输的 `pgps-opsd-data-v1` 数据包放在目标磁盘上。仓库的 `scripts/export_data.py` 可从旧服务器导出此包，见 [数据迁移](docs/DATA_MIGRATION.md)。
- 模型、数据和环境优先放目标机器的本地NVMe盘，实验输出可放目标存储。这里不绑定源服务器路径。

使用已安装的Hugging Face客户端下载模型的示例：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /your/local-disk/models/Qwen3-VL-8B-Instruct
```

也可以用其他下载环境中的ModelScope；不必为下载工具修改锁定的训练依赖。

### 4. 配置与CPU准备

```bash
cp configs/local.example.yaml configs/local.yaml
# 编辑 paths.model、paths.data、paths.output、experiment.name 和六张GPU编号
python scripts/prepare.py --config configs/local.yaml
python scripts/preflight.py --config configs/local.yaml
```

`configs/local.yaml` 被Git忽略。W&B默认 `offline`；若要在线记录，先在目标服务器执行 `wandb login`，再设置 `wandb.mode: online` 和自己的 `entity`。不要把密钥写入配置或仓库。

模型准备会使用固定的 tokenizers 0.22.2 将官方下载文件恢复为旧训练保存的 tokenizer 格式，并逐字节核对旧 SHA；原始模型不改写。

数据准备会核对原parquet、全部训练图片SHA、原题/原解析、学生输入、行序、补齐及测试集隔离。旧实验的训练图与 `Diagram_Visual` 不完全相同，不能自行替换。生成的parquet仅重写图片路径，并按配方追加教师约束。

### 5. 两步验证，然后正式训练

```bash
# 六张卡必须空闲；先完成12+6条样本的两次真实更新
python scripts/launch.py --config configs/local.yaml --mode smoke

# 六rank检查点CPU合并，再在一张空闲卡做单题真实生成；不调用裁判
python scripts/smoke_check.py --config configs/local.yaml --gpu 6

# 仅验证全部通过才允许正式669步；从原始base重新初始化
python scripts/launch.py --config configs/local.yaml --mode train
```

命令在前台运行；长时间任务可放在自己开启的 `tmux` 会话中。准备代码或检测到空卡均不会自动启动训练。失败后先检查日志；入口拒绝覆盖旧运行，不提供未经验证的自动续训。

详细过程、日志位置和检查项见 [训练说明](docs/TRAINING.md)。

### 6. 训练完成后导出与评测

```bash
python scripts/export.py --config configs/local.yaml --mode train --step 669
```

导出目录为 `<output>/<experiment.name>/run/exports/step_0669/model`；其他已保存步骤同理。把导出路径交给已有的 **Unified Eval** 评测框架。这个仓库不复制旧服务器的完整基线作答/裁判缓存，也不自动启动付费裁判。

后续PGPS正式比较继续沿用：测试1000题、Diagram_Visual、无选项/无解析，temperature=0.7、top_p=0.8、top_k=20、presence_penalty=1.5、repetition_penalty=1、max_tokens=32768、seed=42；截断直接判错，其他回答由同一裁判口径评判。训练图片处理和正式评测图片处理配置不同，应分别保留各自协议。

## 教师约束

`hidden_guidance` 在旧原始解析后追加 [这段文本](configs/prompts/teacher_hidden_guidance.txt)，原题和解析不改写：

> Use the annotations only as hidden guidance to understand and verify the geometric information in the diagram.
>
> Do not mention, quote, cite, or refer to the annotations or their existence! Do not reproduce their headings or raw format!
>
> Present a self-contained solution to the original problem, stating the relevant geometric facts directly in ordinary mathematical language.

对照实验把 `experiment.recipe` 改为 `original`，使用新的实验名重新准备；不要修改已经冻结并启动的实验。

## 已验证与待验证

源码与原六卡记录核对；迁移脚本的CPU测试和数据验证记录见 `provenance/`。旧完整环境没有二进制镜像，恢复版本不等于承诺逐比特复现旧成绩。

**尚未在目标8×H100服务器运行真实训练。** 目标服务器必须完成环境导入、CPU预检、两步真实更新、导出及单题生成后才能开正式实验。H100与A800可能有浮点/内核差异，不承诺完全相同的采样轨迹或准确率。

## 来源

基于固定 Vision-OPD/verl 源码和历史实验修改；保留原Apache-2.0许可证及版权头。见 [来源与修改记录](THIRD_PARTY_NOTICES.md)。数据集和模型分别遵循其原始许可，不由本仓库代码许可替代。
