# 直接读取官方PGPS9K（推荐）

`paths.data`填写官方解压目录即可；支持`train.json`与`Diagram_Visual`同层，或`PGPS9K/train.json`与`Diagram_Visual`同属下载根目录。使用：

```bash
python scripts/run.py --config configs/local.yaml
```

程序只读取8021道训练题，学生使用原题和原图；教师额外使用原始结构/语义解析及所选约束。选项、答案、expression不写入训练样本。存在test.json时检查题号不交叉。按题号数值排序，为六卡分批补一条首题，保持8022行、batch12、669步。

原图直接从Diagram_Visual读取，不复制、不离线缩放，不需要clight_images或旧Parquet。运行时仍由模型processor完成必要的图像处理。每张图的原始尺寸、哈希及输入来源记录在实验审计中；不强制与旧预处理图片的字节匹配。

verl需要的`data/train.parquet`和`data/smoke.parquet`由程序在实验目录自动缓存，无需手动管理。默认保留缓存和审计，便于复查实际输入；原始下载目录不会被改写。`--prepare-only`可只准备和做CPU检查。

以下是**可选的历史图片复现方式**，只有主动使用旧数据包时才适用；新服务器直接用官网下载数据可以忽略。

---

# PGPS 数据迁移

GitHub 仓库存放代码、配置、依赖清单和文档。真实题目及训练图片单独打包传输，模型在新服务器下载；不需要迁移虚拟环境目录、旧权重或全部历史回答。

## 数据包布局

```text
pgps-opsd-data-v1/
  manifest.json
  source/train.parquet       # 原无约束训练文件，字节不改
  train.json                 # 原始官方8,021题
  test.json                  # 原始官方1,000题，用于检查题号无混入
  training_images/*.png      # parquet实际使用的3,776张训练图片
```

`manifest.json` 只记录相对包路径、哈希和验收数据，不记录旧服务器绝对路径。为保留可验证的原始数据，`source/train.parquet` 内仍保留旧图片路径；这是私有数据包的一部分。新机器准备实验时不会访问那些旧路径，而是按已核验的图片身份改写为本机路径。

该包不含模型、裁判记录、历史推理或评测图片。`test.json` 含数据集原始答案等字段，但仅用于核对 split；训练 prompt 逐行构建核验，不将答案或选项加入学生或教师输入。

## 源服务器导出

在装有项目依赖、能够读取原训练数据和图片的机器执行：

```bash
python scripts/export_data.py \
  --source-parquet /source/old-nochoices-data/train.parquet \
  --official-train /source/PGPS9K/train.json \
  --official-test /source/PGPS9K/test.json \
  --image-manifest /source/manifests/training_images.json \
  --output /datasets/pgps-opsd-data-v1
```

上面是需要替换的路径示例。`--image-manifest` 接受旧迁移包 `training_images.json` 的 `files` 清单（每项有 `path` 和 `sha256`）；实际复制位置取自原 parquet，不依赖旧 manifest 中的绝对来源路径。执行只导出数据，不启动训练、GPU服务或裁判。

导出程序固定核验以下原始身份：

| 文件 | SHA256 |
|---|---|
| 原无约束 train.parquet | `9c69e065ec8432f5f66e324d69523c63237d61af9ea07e7d57a43b4e3627d577` |
| 官方 train.json | `dcb397f4f55ed5075c4174fbac304ea4719979217cc561023954b0c1692eb377` |
| 官方 test.json | `682fbfe54f02e3baf05f08194fc6b9ebaca3d190ac856f83c75a10f40264b298` |

图片逐个核对 SHA256，并固定整套文件名→内容哈希的摘要，避免只改 manifest 就接受另一套图片。检查通过后写入私有临时目录，再发布完整目录；已有输出目录拒绝覆盖。迁移时直接复制完整包，保留相对结构即可。

旧大迁移包不是本程序直接接受的 bundle。可在旧服务器用上述命令重新导出：原训练 parquet、官方 JSON 和实际训练图片可以复用，历史 `manifests/training_images.json` 也可复用。不要改名假装符合新 manifest，也不要把旧 Qwen3.5 的 framework、环境或四卡训练配置复制成本项目配置。

## 为什么必须复制实际训练图片

旧成功实验读取的是 `clight_images` 中的 RGB PNG，共 **3,776 张、88,566,205 bytes（约84.5 MiB）**。它们已经过旧数据准备路径处理，不等于原 `Diagram_Visual` 同名文件。

例如 `img_1934.png`：实际训练图为224×192，原图为232×203，像素和字节均不同。迁移不得重新缩放、转码或换用同名原图。程序按字节复制，并拒绝同名但内容不同的图。训练图和以后评测用的 `Diagram_Visual` 应各自存放。

## 新服务器准备实验

将本机配置中的 `paths.data` 指向新 bundle 后，使用项目统一入口：

```bash
python scripts/prepare.py --config /path/to/local.yaml
```

准备入口会调用：

```python
prepare_data(data_root: Path, output_dir: Path, suffix: str) -> dict
```

只核验数据时也可从 Python 调用 `geo_opsd.data.verify_bundle(data_root)`；它只读，不生成新文件。依赖包含 pyarrow，测试使用实际 Arrow 读写，不用模拟 Parquet 代替。

准备阶段检查并保留：

- **8,022行、8,021个题号**，原行顺序和唯一补齐行 `prob_0`；不去重、不排序、不重新补齐。
- 学生精确为一个 user 消息，内容是 `"<image>" + 原始题目`，不插入额外空格、system、选项、标准答案或格式要求。
- 教师精确为同一原题，加原前置说明和原结构/语义解析；解析数组顺序和换行不改。
- `reward_model.ground_truth` 为空，学生和教师图片一致，题号、图号对应，训练/测试题号不交叉。
- 全部训练图片 SHA256、原 Arrow schema（包括 metadata）、全部其他列值不变。

唯一变化是 `images[*].path` 和 `teacher_images[*].path` 改成本机包内路径，以及配方选择的教师末尾约束。`original` 配方不加后缀；`hidden_guidance` 只在原教师正文末尾追加两个 LF 和下面的精确文字：

```text
Use the annotations only as hidden guidance to understand and verify the geometric information in the diagram.

Do not mention, quote, cite, or refer to the annotations or their existence! Do not reproduce their headings or raw format!

Present a self-contained solution to the original problem, stating the relevant geometric facts directly in ordinary mathematical language.
```

约束文件末尾的单个 LF 不进入模型输入。文件含该 LF 的 SHA256 为 `105fba20fecbb4b3f8cbbbfd75464378a6fbfef27bbdeb873702240ea148fdbb`。模块不自行 strip 原题或原解析；传入 suffix 带首尾换行会报错。原始数据身份固定，因此不会在已经加约束的教师文本上再次追加。

输出 `train.parquet`、前18行原序的 `smoke.parquet` 和 `audit.json`。报告记录原始身份、行序摘要、全部图片校验、约束内容及哈希、输出文件哈希，并明确未启动训练。写入后重新读回逐行和 schema 验证。目标 Parquet 因图片绝对路径及 writer 版本改变，文件 SHA 不要求等于源文件；源文件的字节始终保留在 bundle 中。

正式 batch12、669步、最后一批6条不变。新服务器虽然有8张H100，恢复实验仍先使用原实现的**6张训练卡**，不因硬件数量重新补齐行或改变归一化。数据校验通过后仍需执行目标环境、processor、FA2以及两步真实训练检查；纯数据核验不能替代GPU兼容性验证。

## 代码、数据与许可证边界

保留 framework 中的 Apache-2.0 LICENSE、已有版权/归属及适用的 NOTICE；迁移适配的修改应有说明。框架许可证不等于 PGPS 数据、图片和模型的许可证；目前本地包没有足够证据单独确认数据可公开再发布，默认仅在用户自己的服务器间独立迁移。模型按官方来源下载，不随仓库分发权重。

不要将旧迁移包整体提交 GitHub。真实数据和历史 provenance 可能包含用户名、绝对路径、内部服务器或评测引用，应留在私有数据包。仓库不收录密钥、`.env`、`.netrc`、SSH凭据/socket、W&B登录配置、API请求日志、conda目录、wheel、缓存、图片、parquet、模型或历史结果。可提交合成测试 fixture 和不含真实值的配置示例。

## 测试

```bash
python -m unittest discover -s tests -p test_data.py -v
```

测试以小型合成题和真实 Parquet 覆盖：迁移后旧路径不可用、原 schema/顺序/其余列保留、教师只追加后缀、图片篡改及 manifest篡改、答案/提示词混入、补齐错误、同名异图、软链接/目录逃逸、已有输出保护。不会读取模型、调用网络或裁判。
