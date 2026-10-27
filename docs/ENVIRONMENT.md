# 旧 Qwen3-VL-8B 训练环境

这里提供版本清单和安装脚本，不包含虚拟环境、大型 wheel、模型或凭据。默认保持旧六卡实验的训练、推理、数值与图像处理核心，允许辅助工具使用兼容版本，再单独验证目标服务器上的六卡训练。

**新服务器的 H100 尚未实测。** 安装、导入检查通过不能代替真实的模型前向、反向、rollout、权重导出及推理检查。

## 固定版本与来源

| 项目 | 历史版本 |
|---|---|
| Python | 历史为3.12.14；默认接受CPython 3.12的其他补丁版本 |
| PyTorch | 2.8.0，运行时 CUDA 12.8，CXX11 ABI=true |
| Transformers / Tokenizers | 4.57.3 / 0.22.2 |
| vLLM / Triton | 0.11.0 / 3.4.0 |
| FlashAttention 2 | 2.8.3 |
| Ray / xFormers | 2.58.0 / 0.0.32.post1 |
| W&B / TensorBoard | 历史为0.30.0 / 2.20.0；默认允许兼容版本 |

提供两种安装模式：

- **`compatible`（默认）**：`constraints.core.txt`保留78个历史核心版本，包括Torch/vLLM/Transformers/FA2、CUDA库、分布式运行、tokenizer、数值和图像处理等。`requirements.compatible.txt`安装这些组件及实际需要的日志工具，使用pip正常解析依赖。W&B、TensorBoard、tqdm、rich、click、pip等不再锁死到历史单一版本；依赖包允许增减。所有实际版本和相对历史清单的变化均记录。
- **`historical`（显式选择）**：`expected_versions.json`保留全部193个历史包的证据；`requirements.lock.txt`固定其中192个，`bootstrap.lock.txt`固定pip、setuptools和wheel。只有此模式要求完整历史清单一致。用`--profile historical`选择。

两种模式都单独安装同一个SHA256核验的FA2 wheel，执行`pip check`和真实导入检查。默认模式中的辅助版本差异不会直接导致验证失败；核心版本变化、缺失依赖、重复metadata和导入失败仍会报错。实际需要的兼容版本由包自身依赖约束决定，不能因此保证所有未来版本均可运行。

这不是完整的操作系统镜像：驱动、glibc、Conda 的基础系统库、硬件与运行路径仍可能不同。Linux x86_64 需要至少 glibc 2.28 来安装历史 PyTorch wheel；其余 wheel 的要求也必须满足，安装器会在无法匹配时失败。

## 在新服务器安装

先安装 Conda，并把本仓库复制到新服务器。以下路径全部替换为该服务器自己的位置。

推荐显式提供原 FA2 wheel：

```bash
bash environment/install.sh \
  --prefix /your/disk/conda-envs/geometry-opsd-legacy \
  --cache-dir /your/disk/cache/geometry-opsd-install \
  --flash-attn-wheel /your/downloads/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

也可以显式从官方固定版本下载：

```bash
bash environment/install.sh \
  --prefix /your/disk/conda-envs/geometry-opsd-legacy \
  --cache-dir /your/disk/cache/geometry-opsd-install \
  --download-flash-attn
```

使用国内镜像且绕过环境代理：

```bash
bash environment/install.sh \
  --prefix /your/disk/conda-envs/geometry-opsd-legacy \
  --cache-dir /your/disk/cache/geometry-opsd-install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --conda-channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  --no-proxy \
  --flash-attn-wheel /your/downloads/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

不指定 `--no-proxy` 时继承标准 HTTP(S)/ALL 代理环境变量。镜像 URL 不应嵌入账号、口令或 token。GitHub 官方下载不保证在所有网络可达；下载失败时可在其他机器取得同一个 wheel 后传入本地文件，SHA256 必须一致。

默认使用Python 3.12，由Conda选择可用补丁版本；FA2的cp312 ABI要求保持不变。可以显式传`--python-version 3.12.X`固定某个补丁。`historical`默认仍为3.12.14。核心版本若在镜像缺失，安装器停止并显示阶段与日志，不会静默更换核心版本；可以更换镜像。辅助工具与Python补丁的实际差异会写入验证报告。

安装器拒绝操作已有环境。只有带有其创建标记的独立 prefix，才允许用相同参数加 `--resume` 继续。Conda 创建中途失败而 Python 尚不可用时，需要先检查日志，不能用其他现存环境冒充。

从旧脚本升级时，先等正在执行的安装命令退出，不要同时运行两个安装器。更新仓库后保持同一个`--prefix`和`--cache-dir`并加`--resume`，即可继续本脚本创建的环境；不会触碰另一个已有的`verl`环境。无需仅因辅助包版本不同而删除已经下载的环境。

## FA2 wheel 校验

上游：[Dao-AILab / flash-attention v2.8.3](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3)。固定文件信息也保存在 `environment/flash_attn_wheel.json`。

```text
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
bytes: 256040057
sha256: f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb
```

此 SHA256 来自旧实验原始 wheel，并已重新计算验证。安装后的包 metadata 版本为 `2.8.3`；文件名中的 CUDA、Torch、ABI 标记不等同于包版本。

校验失败立即停止，不允许自动替换为另一个 FA2 wheel、较新版本或 SDPA。需要在新 H100 上验证原 wheel 的真实运行，不能仅凭文件名推断它已经适配成功。

## 检查与输出

安装全程隐藏 GPU。每个阶段会实时显示输出和日志位置，包括Conda创建、pip下载、FA2下载和检查；出错会显示当前日志路径。缓存目录保存追加式阶段日志、`installed.freeze.txt`和`verification.json`，不会自动启动训练，也不会配置W&B/API密钥。

激活后重新进行 CPU 导入检查：

```bash
conda activate /your/disk/conda-envs/geometry-opsd-legacy
python environment/verify_environment.py --imports \
  --prefix /your/disk/conda-envs/geometry-opsd-legacy \
  --output /your/disk/audits/environment-cpu.json
python -m pip check
```

在确认可用的 GPU 上，显式检查驱动和 H100 识别情况：

```bash
CUDA_VISIBLE_DEVICES=0 python environment/verify_environment.py \
  --imports --cuda --expect-gpu-regex H100 \
  --output /your/disk/audits/environment-h100.json
```

这里的 `0` 是示例，需要改成实际允许使用的卡。CUDA 检查可能初始化上下文，但不会加载模型或执行训练；它只确认设备发现、能力信息和运行时加载。**FA2 kernel、vLLM rollout、六卡通信与完整训练仍须通过项目的独立 smoke test。**

默认模式使用pip依赖解析器和核心约束；完整`historical`模式仍以`--no-deps`安装历史快照。若`pip check`报错，保留日志并诊断具体冲突，不要取消核心版本约束或整体升级。训练框架源码由项目另行安装，继续使用已审计代码和`--no-deps`，不能从上游main替换。安装通过仅说明当前依赖可解析、版本和导入检查通过；H100真实训练尚须独立验证。
