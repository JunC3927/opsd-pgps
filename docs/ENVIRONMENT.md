# 旧 Qwen3-VL-8B 训练环境

这里提供可重建的版本清单和安装脚本，不包含虚拟环境、大型 wheel、模型或凭据。目标是恢复旧六卡实验的软件实现，再单独验证目标服务器上的六卡训练。

**新服务器的 H100 尚未实测。** 安装、导入检查通过不能代替真实的模型前向、反向、rollout、权重导出及推理检查。

## 固定版本与来源

| 项目 | 历史版本 |
|---|---|
| Python | 3.12.14，CPython 3.12 ABI |
| PyTorch | 2.8.0，运行时 CUDA 12.8，CXX11 ABI=true |
| Transformers / Tokenizers | 4.57.3 / 0.22.2 |
| vLLM / Triton | 0.11.0 / 3.4.0 |
| FlashAttention 2 | 2.8.3 |
| Ray / xFormers | 2.58.0 / 0.0.32.post1 |
| W&B / TensorBoard | 0.30.0 / 2.20.0 |

`environment/expected_versions.json` 记录全部 **193 个**历史运行包。它们与升级前保存的环境清单逐项一致。`requirements.lock.txt` 固定其中 192 个包；FA2 由独立的原始 wheel 安装，防止自动换版本或触发源码编译。`bootstrap.lock.txt` 还固定旧 overlay 使用的 pip 25.0.1，以及 setuptools、wheel。

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

默认使用 Python 3.12.14。如果某个历史版本在镜像上缺失，安装器会停止，不会自行改成最新版。可换镜像，或明确指定 `--python-version 3.12.X` 使用同一 CPython ABI 的其他补丁版本；验证报告会记录该差异，此时不能宣称 Python 与历史实验完全一致。

安装器拒绝操作已有环境。只有带有其创建标记的独立 prefix，才允许用相同参数加 `--resume` 继续。Conda 创建中途失败而 Python 尚不可用时，需要先检查日志，不能用其他现存环境冒充。

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

安装全程隐藏 GPU。缓存目录保存每个阶段的日志、`installed.freeze.txt` 和 `verification.json`，不会自动启动训练，也不会配置 W&B/API 密钥。

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

安装脚本使用 `--no-deps` 和固定完整依赖清单；若 `pip check` 报错，保留日志并诊断原因，不要通过取消版本锁或升级最新包来“修复”。训练框架源码由项目另行安装，必须使用已审计的代码和 `--no-deps`，不能临时从上游 main 取代。
