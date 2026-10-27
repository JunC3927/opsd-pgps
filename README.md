# PGPS Geometry OPSD

将已核对的 **Qwen3-VL-8B-Instruct、原六卡 Vision-OPD/verl 实现**迁移到独立 Linux 服务器。提供两个配方：原教师提示词 `original`，以及只给教师追加最终约束的 `hidden_guidance`。

直接读取从官方解压的PGPS9K目录（`train.json`＋`Diagram_Visual`）。无需旧训练图包、图片转换或手动生成Parquet；填写模型、数据和输出路径后，一条命令完成检查与训练。GitHub仓库不包含数据或模型。

## 固定的实验设置

| 项目 | 配置 |
|---|---|
| 模型 | 原始 Qwen3-VL-8B-Instruct；学生与冻结教师相同大小 |
| 训练数据 | PGPS9K 8021题、8022行；保留旧行序与原补齐行 |
| 学生输入 | 官方Diagram_Visual原图＋原题，无选项、解析或答案 |
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
environment/        核心版本约束、历史完整清单、FA2校验、安装/验证脚本
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

默认采用 `compatible` 模式：torch 2.8.0、transformers 4.57.3、vLLM 0.11.0、FA2 2.8.3 及相关训练/数值/图像组件保持旧版本；W&B、TensorBoard、tqdm、rich、click等辅助工具由pip选择兼容版本，不再逐个强求历史版本。Python要求3.12 ABI，默认不固定补丁号。安装时实时显示阶段与输出，同时保存日志。

要恢复旧的全部依赖版本限制，可显式加 `--profile historical`。默认模式也会记录实际版本及历史差异；仍需通过H100真实训练验证，不能视为完整环境逐字节复现。完整说明见 [环境说明](docs/ENVIRONMENT.md)。

### 3. 下载模型和官方PGPS9K

- 下载官方 `Qwen/Qwen3-VL-8B-Instruct` 到新服务器。
- 下载并解压官方PGPS9K，直接使用其中的 `Diagram_Visual`。**不需要旧服务器的数据包、clight_images或train.parquet。**

支持这两种常见布局；`paths.data`指向下面的PGPS目录：

```text
PGPS9K/                         # 或：PGPS9K/
  Diagram_Visual/*.png         #       Diagram_Visual/*.png
  PGPS9K/train.json            #       train.json
  PGPS9K/test.json             #       test.json
```

也支持将`paths.data`指向内层存放train.json的PGPS9K目录。`test.json`存在时用于检查训练/测试题号不重叠；测试题不会作为训练样本。缺少`Diagram_Visual`会报错，不会静默替换为`Diagram`。

### 4. 填写路径和GPU编号

```bash
cp configs/local.example.yaml configs/local.yaml
```

编辑`configs/local.yaml`中的三个路径：

```yaml
paths:
  model: /your/disk/models/Qwen3-VL-8B-Instruct
  data: /your/disk/datasets/PGPS9K
  output: /your/disk/experiments
```

同时确认`experiment.name`是新实验名，`experiment.gpus`为六张允许使用的卡。`hidden_guidance`给教师追加最终约束，`original`用于无约束对照。W&B默认offline；在线记录需先`wandb login`再设置自己的entity及online模式。

### 5. 一条命令运行

```bash
python scripts/run.py --config configs/local.yaml
```

这条命令**会启动GPU训练**：自动读取官方JSON与原图，检查数据/配置，完成六卡两步试跑、模型导出和单题真实生成，全部通过后从原始base开始正式669步训练。试跑和正式训练使用相同六张卡；单题生成默认复用试跑结束后释放的第一张卡，不要求额外空卡。可在`tmux`中运行。

如果只想先检查数据与配置：

```bash
python scripts/run.py --config configs/local.yaml --prepare-only
```

图片原地读取，不复制、不离线缩放、不另存RGB图片。现有verl训练器需要的两个小型Parquet缓存由程序自动生成在实验输出目录；它们只保存提示词、图片路径等字段，无需手动准备或迁移。原图在读取时转RGB，再由模型processor按既定设置处理。相比历史clight_images，这是一项已明确记录的图像输入变化。

已有成功阶段会核对后复用；中止/失败训练不会被自动重跑或覆盖。原有分步prepare/preflight/smoke/train入口仍保留，详见[训练说明](docs/TRAINING.md)。

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
