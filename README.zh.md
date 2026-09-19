# Dense Dump Codec：使用说明

Dense Dump Codec（DDC）用于压缩密集的 GRMHD 时间序列，支持随机访问、
原格式文件恢复和慢光辐射转移所需的原生数组读取。

## 1. 方法与适用范围

将连续输出划分为 GOP：保留两端原始精确锚点，按实际时间预测中间状态，
再保存经过量化和压缩的残差。读取某个状态时只需对应锚点与局部通道块。
算法是一般性的，但本版文件适配器针对 KHARMA/Parthenon PHDF；其他格式需要适配。

推荐配置为 rho/u/速度/B = **8/8/5/16 bit**。rho、u 在自然对数空间处理；
只有 u 的尺度进一步按 `(phi, theta, r)=(8,16,32)` 网格块细分。
超出尺度范围的残差以未量化 float32 值保存。
旧版 MeshBlock 尺度、局部 u8、普通文件恢复和原生数组读取均支持。

时间间隔由输入序列决定。GOP 默认跨度为 25 个输出步，
输入间隔为 0.1M 时对应约 2.5M。

## 2. 安装与快速开始

环境要求：Linux、Python 3.11+，核心依赖为 NumPy 和 h5py。
如使用 LZ4 空间工作缓存，还需要系统 `liblz4`；可选 C/CUDA 加速另需编译器。

**第一步：安装。** 创建独立 Python 环境，安装程序及测试依赖。
`-e` 表示直接使用当前源码目录；`[dev]` 额外安装测试、打包工具。
如果只使用编解码、不运行测试，可将最后一行改为 `python -m pip install -e .`。

```bash
git clone https://github.com/zelinzh/dense-dump-codec.git
cd dense-dump-codec
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

**第二步：检查安装（可选）。** 运行自动测试，确认编解码、随机读取等功能正常。

```bash
python -m pytest -q
```

**第三步：生成小型测试数据。** 没有真实模拟文件也能试用：下面生成 51 个
合成 PHDF 状态，放在 `demo/raw/`。请使用尚不存在的 `demo/` 目录开始本例。

```bash
python examples/make_synthetic_sequence.py --output-dir demo/raw
```

**第四步：编码并检查完整性。** 编码为两个 GOP，保留三个原始锚点。
`verify --sha256` 检查档案、锚点和记录的校验值。编码后保留 `demo/raw/` 中的原文件。

```bash
ddc encode --segment-dir demo/raw --output-dir demo/ddc
ddc verify --manifest demo/ddc/sequence_manifest.json --sha256
```

**第五步：解码为原格式文件。** 恢复序号为 13 的状态，写成常规 PHDF 文件，
可交给读取该格式的分析程序。无需先恢复所有其他状态。

```bash
ddc decode --manifest demo/ddc/sequence_manifest.json --sequence 13 \
  --output-phdf demo/reconstructed.out0.00013.phdf
```

另一种读取方式是不生成 PHDF，直接在内存中取得数组。下面的可选示例读取同一状态，
并打印时间、字段形状等信息，适合后续程序直接接入 DDC。

```bash
python examples/read_native_frame.py --manifest demo/ddc/sequence_manifest.json --sequence 13
```

**第六步：比较误差（可选）。** 以合成原始数据为真值，比较 DDC 重建误差与
每隔 2、5、10 帧保留原始数据后进行线性时间插值的误差，将统计写入 `demo/comparison.json`。
运行比较需要保留完整原始真值。

```bash
ddc compare --truth-dir demo/raw --codec-manifest demo/ddc/sequence_manifest.json \
  --datasets prims.rho,prims.u,prims.uvec,prims.B \
  --raw-strides 2,5,10 --output-json demo/comparison.json
```

有自己的数据时可跳过合成数据生成，直接使用下面的编码、解码命令。

## 3. 正式编码

```bash
ddc encode --segment-dir /data/run --output-dir /data/ddc \
  --profile local-u8 --keyframe-stride 25 --archive-workers 4
```

- 输入文件名为 `*.out0.<整数>.phdf`，时间取自 `Info/Time`。
- 输入含 `prims.rho`、`prims.u`、`prims.uvec`、`prims.B`，一般为 float32。
- 标量数组为 `(MeshBlock, phi, theta, r)`，矢量多一个分量轴。
- 空间维度须能被 tile 整除。
- `local-u8` 是默认 preset；`legacy-u8` 是旧 u 尺度；`custom` 用低层参数。
- 显式选项可以覆盖 preset，实际配置写入每个 GOP 和 manifest。
- `configs/local_u8.json` 是配置说明，不会被命令自动读取。

完整序列由 `sequence_manifest.json`、GOP 容器和输入目录中的原始锚点共同组成，
三者需一同保留。移动数据时需维护 manifest 和容器中的路径引用。
编码默认保留源文件；模拟重启使用独立保存的 RHDF。

## 4. 解码回原始文件格式

只需 manifest、所需 GOP 容器及其原始锚点，就能恢复指定状态；被编码的中间原文件
不参与解码。路径必须与档案中的引用一致。下面以原文件序号 `13` 为例：

```bash
ddc decode --manifest /data/ddc/sequence_manifest.json --sequence 13 \
  --output-phdf /data/restored/torus.out0.00013.phdf
```

- `--sequence` 是原始文件名中的输出序号，不是物理时间，也不一定从 0 开始。
  manifest 的 `frames` 列表记录了可用的 `sequence` 和 `time`。
- `--output-phdf` 指定恢复文件的路径，父目录会自动创建。默认拒绝覆盖已有文件；
  确实需要覆盖时才添加 `--overwrite`。请使用独立输出目录，避免覆盖真值或锚点。
- 默认恢复编码时选定的全部字段；推荐配置包含密度、内能、速度和磁场。
- 中间帧由预测和量化残差重建，含量化误差；原始锚点原样复制。
- 中间帧文件以锚点为结构模板，写回已编码字段和物理时间。
  未编码的诊断字段沿用模板值，而非目标时刻的结果；循环计数由锚点插值。
  配套 XDMF 需另行生成。

恢复多个状态或整段序列的示例见 [批量恢复说明](docs/WORKFLOW.md#restore-conventional-files)。
若下游程序支持 DDC 原生读取，则不必将整段序列展开成 PHDF，可节省临时磁盘空间。

## 5. 流式编码与读取

```bash
ddc watch --segment-dir /data/run --output-dir /data/ddc-stream \
  --expected-frame-count 1001 --keyframe-stride 25 --profile local-u8
```

watcher 等待 GOP 与后继状态就绪后编码；最后一段另做稳定性检查。
中断后用相同命令恢复，不应同时启动两个写入同一目录的 watcher。
默认不删除任何中间原始数据；只有显式启用删除策略才会在验证和提交后删除。
流式编码需要为临时密集输出预留工作空间。

Python 原生读取示例见 [API](docs/API.md)。KPolaris 的外部编译、
LZ4 工作缓存、半径裁剪与紧凑传输见 [集成说明](docs/KPOLARIS.md)。
工作缓存可在使用结束后删除，需要时重新生成。

## 6. 许可

[BSD-3-Clause](LICENSE)。

## 7. 引用

如果在研究中使用 DDC，请引用本软件：

```bibtex
@software{zhang_dense_dump_codec,
  author  = {Zhang, Zelin},
  title   = {{Dense Dump Codec}},
  year    = {2026},
  version = {0.4.0rc1},
  url     = {https://github.com/zelinzh/dense-dump-codec}
}
```

机器可读的引用信息见 [CITATION.cff](CITATION.cff)。
