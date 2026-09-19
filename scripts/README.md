# Script guide / 脚本使用导航

For normal use, start with `ddc encode`, `ddc verify`, and `ddc decode`.
These commands use the implementations listed below.

日常用法见根目录的中文说明，以下列出命令入口和辅助工具。

## Main commands / 主要命令

| Script | Preferred command | Purpose / 用途 |
|---|---|---|
| `encode_dense_sequence.py` | `ddc encode` | Encode an existing sequence / 编码已有序列 |
| `decode_dense_sequence.py` | `ddc decode` | Restore one state as PHDF / 将一个状态恢复为原格式 |
| `verify_ddc_sequence.py` | `ddc verify` | Check required files and integrity / 文件与完整性检查 |
| `watch_dense_codec.py` | `ddc watch` | Encode completed GOPs during a run / 流式编码和恢复 |
| `compare_dense_cadences.py` | `ddc compare` | Compare field errors against retained truth / 对比真值和插值误差 |

The `ddc encode` and `ddc watch` commands apply the `local-u8` preset by default.
Direct script calls use their own defaults; see each script's `--help`.

## Internal helpers / 内部实现

| Script | Purpose / 用途 |
|---|---|
| `prototype_dump_codec.py` | GOP prediction, quantization and shared field utilities / 编码核心及共享工具 |
| `decode_dump_codec.py` | GOP-to-array/file reconstruction used by sequence readers / 序列读取器依赖的 GOP 解码实现 |

## Optional acceleration and GRRT / 可选加速与 GRRT 接口

| Script | Command or role / 命令或用途 |
|---|---|
| `prepare_ddc_working_cache.py` | `ddc prepare-cache`: reusable spatial working cache / 空间工作缓存 |
| `serve_ddc_native.py` | `ddc serve`: native-array service / 原生数组服务 |
| `build_ddc_predict_kernel.py` | `ddc build-predictor`: optional CPU kernel / 可选 CPU 加速核 |
| `run_kpolaris_ddc_window.py` | `ddc run-kpolaris`: external KPolaris integration / 调用外部 KPolaris |
| `run_kpolaris_raw_window.py` | External KPolaris with original files / 原始文件对照入口 |
| `materialize_ddc_for_kpolaris.py` | On-demand PHDF materialization with bounded prefetch / 按需恢复文件 |
| `serve_ddc_for_kpolaris.py` | Compatibility service for materialized files/native arrays / 兼容服务 |

KPolaris setup and transport options are described in
[the integration guide](../docs/KPOLARIS.md).

## Validation, archive utilities and figures / 检查、档案工具与绘图

| Scripts | Purpose / 用途 |
|---|---|
| `validate_phdf_integrity.py` | Check original PHDF files / 检查输入文件完整性 |
| `repack_ddc_archive.py`, `repack_ddc_sequence.py` | Change a residual archive's lossless packing / 残差档案重打包 |
| `repack_phdf_keyframe.py`, `repack_sequence_keyframes.py` | Optional lossless anchor archives; not the default retained-original-anchor workflow / 可选锚点打包，非默认流程 |
| `rebase_codec_comparison_storage.py` | Update comparison storage accounting after verified repacking / 重打包后的存储统计 |
| `plot_ddc_method_document.py`, `plot_ddc_gop_structure.py` | Method diagrams, requiring the `figures` extra / 方法示意图，不参与编解码 |
