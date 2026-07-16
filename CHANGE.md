# CHANGE.md

Twinkle 在 **Moore Threads MUSA** 环境下的变更记录。  
目标：跑通 `cookbook/client/twinkle/self_host/self_cognition.py`（Transformers + LoRA + Ray Serve）。

> 原则：不覆盖、不重装现有 `torch` / `torch_musa`；Platform 仿 NPU 最小实现。  
> 详细过程备忘见同目录 `musa_twinkle.md`；本文按 **修改原因 → 修改前 → 修改后** 重组。

---

## 0. 环境基线（修改所依据）

| 组件 | 版本 | 备注 |
|------|------|------|
| Python | 3.10.12 | |
| torch / torch_musa | 2.7.1 | `torch.musa.is_available() == True` |
| torchvision | 0.22.1+32091f2 | MUSA 配套 |
| accelerate | 1.12.0 | 已识别 MUSA |
| transformers | 4.50.2 | |
| numpy / scipy | 1.26.0 / 1.10.1 | 不可强升到 numpy 2.x |
| 硬件 | 8× MTT S5000 | |

---

## 变更 1：`pyproject.toml` 依赖对齐

### 修改原因

- 仓库原约束与本机 MUSA 栈冲突：`requires-python>=3.11`、`numpy>=2.0` 等。
- 裸装 extras 可能拉到 CUDA 官方 `torch`，破坏 `torch_musa`。
- 需把依赖钉到本机实测版本，保证可复现、可安装。

### 修改前

```toml
requires-python = ">=3.11,<3.13"
dependencies = [
  "numpy>=2.0.0,<2.3.0",
  "datasets",
  "omegaconf>=2.3.0,<3.0.0",
  "fastapi",
  "modelscope[framework]>=1.34.0",
  "safetensors",
  "peft>=0.11.0,<=0.19.0",
  "transformers",
]

[project.optional-dependencies]
transformers = [
  "accelerate",
  "torch>=2.6.0,<3.0.0",
  "torchvision",
]
ray = ["ray[serve]"]
```

### 修改后

路径：`pyproject.toml`

| 字段 | 修改后 |
|------|--------|
| `requires-python` | `>=3.10,<3.13` |
| `numpy` | `==1.26.0` |
| `scipy` | `==1.10.1`（新增） |
| `datasets` | `==4.4.2` |
| `omegaconf` | `==2.3.0` |
| `fastapi` | `==0.116.1` |
| `modelscope[framework]` | `==1.36.3` |
| `safetensors` | `==0.6.2` |
| `transformers` | `==4.50.2` |
| `peft` | `==0.18.0` |
| `accelerate` | `==1.12.0` |
| `torch` | `==2.7.1` |
| `torchvision` | `==0.22.1` |
| `ray` | `ray[serve]==2.55.1` |

说明：`torch_musa`、`pydantic` 不写入主 pin（官方栈 / Ray 传递依赖）。

---

## 变更 2：安装 `peft` / `ray[serve]`

### 修改原因

- `self_cognition` 训练路径需要 peft（LoRA）。
- Twinkle Server 需要 Ray Serve。
- 需在不触碰 `torch_musa` 的前提下补齐二者。

### 修改前

- `peft`：未安装  
- `ray`：未安装  
- `pydantic`：`2.11.7`

### 修改后

执行：

```bash
pip install 'peft==0.18.0' 'ray[serve]==2.55.1'
```

| 项 | 结果 |
|----|------|
| peft | `0.18.0`（已装；但受 TE 影响，当前 **import 仍可能失败**，见「已知问题」） |
| ray | `2.55.1` |
| torch / torch_musa / numpy / scipy | **未变** |
| pydantic | `2.11.7` → `2.13.4`（Ray Serve 排除 2.10/2.11，连带升级；对 torch_musa 无影响） |

---

## 变更 3：新增 MUSA Platform

### 修改原因

- Twinkle 原仅支持 GPU / NPU / MPS，**无 MUSA**。
- 本机无 `nvidia-smi` 时会默认落到 `cuda`，导致 `.to('cuda:0')` 失败。
- 需让 `device_type: musa`、Ray 自定义资源 `MUSA`、设备串 `musa:0` 生效。

### 修改前

- 平台文件：仅 `gpu.py` / `npu.py` / `mps.py` / `base.py`
- `get_platform_names()` → `['GPU', 'NPU', 'MPS']`
- `get_platform('musa')` → `ValueError: Unsupported platform`
- 自动探测：`npu-smi` → NPU；`nvidia-smi` → GPU；否则默认 GPU

### 修改后

| 操作 | 路径 |
|------|------|
| 新增 | `src/twinkle/utils/platforms/musa.py` |
| 修改 | `src/twinkle/utils/platforms/base.py` |
| 修改 | `src/twinkle/utils/platforms/__init__.py` |
| 修改 | `src/twinkle/utils/__init__.py` |

`MUSA` 关键 API：

| 方法 | 返回值 |
|------|--------|
| `visible_device_env()` | `MUSA_VISIBLE_DEVICES` |
| `device_prefix()` | `musa` |
| `get_local_device(idx)` | `musa:{idx}` |
| `device_backend()` | `mccl` |

自动探测改为：NPU → **MUSA（优先于 CUDA）** → GPU → MPS。

验证：

```text
auto MUSA musa MUSA_VISIBLE_DEVICES mccl
local musa:0
```

---

## 变更 4：`server_config.yaml`（Transformers Server）

### 修改原因

- 原配置写死 `device_type: cuda`，无法调度 MUSA。
- `self_cognition.py` 只训练/保存，不需要 sampler，关掉可省卡、避 vLLM。
- 本地已有模型权重，改用本地路径避免重复下载。

### 修改前

- model / mesh：`device_type: cuda`
- `model_id: "ms://Qwen/Qwen3.5-4B"`
- sampler 整段启用（cuda）
- 备份：无

### 修改后

路径：`cookbook/client/server/transformer/server_config.yaml`  
备份：`server_config.yaml.cuda.bak`

| 项 | 修改后 |
|----|--------|
| model `device_type` | `musa` |
| mesh `device_type` | `musa` |
| `model_id` | `/data/wangmx/Qwen3.5-4B` |
| `ranks` / `nproc` / `dp_size` | `1`（单卡首跑） |
| sampler | **整段注释掉** |
| processor | 仍为 `CPU` |
| `supported_models` | 仍为 `Qwen/Qwen3.5-4B`（与客户端一致） |

---

## 变更 5：`run.sh`（Ray 启动）

### 修改原因

- 原脚本用 `CUDA_VISIBLE_DEVICES` + `--num-gpus`，只认 CUDA。
- MUSA 需用 `MUSA_VISIBLE_DEVICES` + Ray 自定义资源 `{"MUSA":N}`（与 `Platform.__name__` 一致）。

### 修改前

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --port=6379 --num-gpus=4 ...
CUDA_VISIBLE_DEVICES=4,5,6,7 ray start --address=127.0.0.1:6379 --num-gpus=4
CUDA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0
python server.py
```

### 修改后

路径：`cookbook/client/server/transformer/run.sh`  
备份：`run.sh.cuda.bak`

```bash
# 要点：
# MUSA_VISIBLE_DEVICES=0
# --num-gpus=0
# --resources='{"MUSA":1}'
# 另有 CPU-only worker；启动前 ray stop --force
```

单卡最小启动；多卡示例写在脚本注释中。

> 后续因 PID 配额问题又做了加固，见 **变更 7**。

---

## 变更 6：editable 安装 twinkle

### 修改原因

- 未安装时 `import twinkle` 失败（除非手动 `PYTHONPATH=src`）。
- 起 Server / 跑客户端都需要能稳定 import。
- 必须 `--no-deps`，避免解析依赖覆盖 `torch_musa`。

### 修改前

- `twinkle-kit`：未安装  
- `import twinkle`：失败（无 PYTHONPATH 时）

### 修改后

执行：

```bash
cd /data/wangmx/twinkle
pip install -e . --no-deps
```

| 项 | 结果 |
|----|------|
| 安装内容 | 仅 `twinkle-kit-0.3.0` editable |
| torch / torch_musa | **未变** |
| `import twinkle` | 指向 `/data/wangmx/twinkle/src/twinkle/...` |

---

## 变更 7：`run.sh` 缓解容器 PID / pthread 耗尽

### 修改原因

- 首次 `bash run.sh` 起服时出现：
  `pthread_create failed: Resource temporarily unavailable`
- 实测容器 cgroup `pids.max≈9830`，`pids.current` 已接近上限；dmesg 有  
  `fork rejected by pids controller in ... docker-...scope`
- 机器/容器对外显示 **CPU=255**，Ray 会 `num_prestart_python_workers=255`，预拉大量 worker，再叠加 OpenBLAS/OMP 多线程，PID 迅速打满
- 方案：启动前强制停干净旧 Ray；限制 BLAS/OMP 线程数；限制 Ray `--num-cpus`（默认 8）

### 修改前（变更 5 之后、加固之前）

```bash
# 仅有 MUSA 资源与 ray stop，未限制线程 / CPU 数量
export RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES=1
ray stop --force 2>/dev/null || true

MUSA_VISIBLE_DEVICES=0 ray start --head --port=6379 --num-gpus=0 \
  --resources='{"MUSA":1}' \
  --disable-usage-stats --include-dashboard=false

MUSA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0
python server.py
```

### 修改后

路径：`cookbook/client/server/transformer/run.sh`

**方案 1 — 启动前清理：**

```bash
ray stop --force 2>/dev/null || true
sleep 2   # 等待 cgroup 回收 PID
```

**方案 2 — 限制原生线程池：**

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAY_worker_maximum_startup_concurrency=8
```

**配套（否则仍按 255 核预拉 worker）：**

```bash
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"

MUSA_VISIBLE_DEVICES=0 ray start --head --port=6379 --num-gpus=0 \
  --num-cpus="${RAY_NUM_CPUS}" \
  --resources='{"MUSA":1}' \
  --disable-usage-stats --include-dashboard=false

MUSA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0 \
  --num-cpus="${RAY_NUM_CPUS}"
```

仍可用环境变量覆盖 CPU 数，例如：`RAY_NUM_CPUS=4 bash run.sh`。

---

## 变更 8：peft 降级避开坏 TE（环境操作）

### 修改原因

- peft `0.19.0` 在 import 时会探测并加载 `transformer_engine`
- 本机 MT-TE `2.0.0+e73781e` 与当前 `torch_musa` 2.7.1 ABI 不匹配，导致 `import peft` 失败
- `0.18.x` 无 TE 探测逻辑，可在不卸载 TE 的情况下恢复 LoRA

### 修改前

- peft：`0.19.0`（已装但无法 import）

### 修改后

```bash
pip install 'peft==0.18.1'
```

- peft：`0.18.1`，`from peft import LoraConfig` 正常  
- TE 仍保留在环境中（本路径 HF+LoRA 不使用）  
- 注：`pyproject.toml` 若仍写 `peft==0.19.0`，与环境不一致时可再改 pin

---

## 已知问题 / 后续

| 项 | 状态 |
|----|------|
| peft × TE | 已用 peft 0.18.1 **绕开**；TE 本体仍坏，Megatron/FP8 以后再处理 |
| 起 Server | 需用加固后的 `run.sh` 重试；若仍顶 PID，提高 Docker `--pids-limit` 或 `RAY_NUM_CPUS=4` |
| 跑 `self_cognition.py` | Server `:8000` 正常后再跑 |

---

## 变更 8：`tinker` 适配 Python 3.10

### 修改原因

Twinkle Gateway 硬依赖 `from tinker import types`；官方 `tinker==0.14.0` 要求 Python ≥3.11，本机 3.10 无法直接安装。

### 修改范围（源码树，非 twinkle 仓内）

路径：`/data/wangmx/tinker-0.14.0`  
完整前后对比见：**`/data/wangmx/tinker-0.14.0/ADAPT_PYTHON310.md`**

| 改动 | 摘要 |
|------|------|
| `requires-python` | `>=3.11` → `>=3.10` |
| `request_error_category.py` | `StrEnum` 3.10 回退实现 |
| `api_future.py` | `asyncio.timeout` → 3.10 用 `wait_for` |
| 安装 | `pip install -e ... --no-deps` + 补 `distro`/`h2` |

### 验证

`from tinker import types` 成功；`torch 2.7.1` / `torch.musa` 未破坏。

---

## 变更 9：`self_cognition.py` 数据集 / template / model 改本地路径

路径：`cookbook/client/twinkle/self_host/self_cognition.py`

### 修改原因

跑脚本时在加载 `ms://swift/self-cognition` 失败：

```text
RuntimeError: Failed to import modelscope.msdatasets ...
No module named 'addict'
```

根因：`modelscope` 核心包未带 `datasets`/`framework` extras，`MsDataset` 依赖 `addict` 等。  
本机已有完整本地资源，改为本地读取可绕开 ModelScope 拉数/拉 tokenizer，也避免继续补一串 extras。

本地资源核对：

| 资源 | 路径 | 结论 |
|------|------|------|
| 数据集 | `/data/wangmx/self-cognition/self_cognition.jsonl` | 108 条，字段 `query`/`response`/`tag`，含 `{{NAME}}`/`{{AUTHOR}}`，符合 `SelfCognitionProcessor` |
| 模型 | `/data/wangmx/Qwen3.5-4B` | 含 `config.json` / tokenizer / safetensors；Server 侧已用该路径 |

注意：`DatasetMeta` 传**目录**会因 `listdir()[0]` 扩名误判失败；必须传 **jsonl 文件**。

### 修改前

```python
base_model = 'Qwen/Qwen3.5-4B'
...
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=512)
...
model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')
...
model.set_template('Qwen3_5Template')
```

### 修改后

```python
base_model = 'Qwen/Qwen3.5-4B'  # 仅作 Server 路由名，对应 /api/v1/model/Qwen/Qwen3.5-4B
local_model = '/data/wangmx/Qwen3.5-4B'
local_dataset = '/data/wangmx/self-cognition/self_cognition.jsonl'
...
dataset = Dataset(dataset_meta=DatasetMeta(local_dataset, data_slice=range(500)))
dataset.set_template('Qwen3_5Template', model_id=local_model, max_length=512)
...
model = MultiLoraTransformersModel(model_id=base_model)  # 不可填本地目录，否则 URL 错乱
...
model.set_template('Qwen3_5Template', model_id=local_model)
```

### 设计说明（为何 model 客户端仍用逻辑名）

| 用法 | 应填什么 | 原因 |
|------|----------|------|
| `DatasetMeta(...)` | 本地 jsonl | `os.path.exists` → `datasets.load_dataset`，不走 ModelScope |
| `dataset.set_template(..., model_id=)` | 本地模型目录 | Processor 侧 `HubOperation.download_model` 遇本地路径直接返回 |
| `MultiLoraTransformersModel(model_id=)` | `Qwen/Qwen3.5-4B` | 客户端拼 URL：`/model/{model_id}/twinkle`，须与 Server `route_prefix` 一致；权重已在 Server 的 `model_id: "/data/wangmx/Qwen3.5-4B"` 加载 |
| `model.set_template(..., model_id=)` | 本地模型目录 | 覆盖客户端默认的逻辑名，避免 Server 再去 Hub 取 tokenizer |

### 未改项

- 未安装 `addict` / 未重装 `modelscope[datasets]`（本地路径方案下不需要）。
- Server `server_config.yaml` 此前已指向本地模型，本次未再改。

---

## 变更 10：升级 `transformers` 至 5.2.0（支持 Qwen3.5）

### 修改原因

本地加载 `/data/wangmx/Qwen3.5-4B` 时，`transformers==4.50.2` 不认识 `model_type=qwen3_5`：

```text
ValueError: The checkpoint ... has model type `qwen3_5` but Transformers does not recognize this architecture.
```

经核对：4.57.x 仍无 `qwen3_5`；**≥5.2.0** 的 wheel 才包含该架构。不采用 4.57.0（无效且 yanked）。

### 修改前 → 修改后（实测）

| 包 | 修改前 | 修改后 |
|----|--------|--------|
| transformers | 4.50.2 | **5.2.0** |
| tokenizers | 0.21.4 | **0.22.2** |
| huggingface-hub | 0.34.4 | **1.23.0** |
| hf-xet | 1.1.9 | **1.5.1** |
| click | 8.2.1 | **8.4.2** |
| typer / typer-slim / shellingham / annotated-doc | 无 | 新装（CLI 用，训练路径不依赖） |
| torch / torch_musa / peft / accelerate / numpy / safetensors | 2.7.1 / 0.18.1 / 1.12.0 / 1.26.0 / 0.6.2 | **未变** |

安装命令：

```bash
pip install 'transformers==5.2.0' --upgrade-strategy only-if-needed
```

`pyproject.toml` 同步：`transformers==5.2.0`，`peft==0.18.1`。

### 验证结果

- `torch.musa.is_available() == True`
- `qwen3_5 in CONFIG_MAPPING`
- `AutoConfig.from_pretrained('/data/wangmx/Qwen3.5-4B').model_type == 'qwen3_5'`
- `AutoTokenizer.from_pretrained(...)` 成功
- `Qwen3_5ForConditionalGeneration` 可从 `transformers` 取到
- `peft` / `huggingface_hub.hf_api.api` 仍可 import

### 说明

- pip 会提示 `twinkle-kit` 元数据与旧 pin 冲突；已用上述 `pyproject.toml` 对齐消除后续警告（需重新 `pip install -e . --no-deps` 才会刷新已安装元数据）。
- 保持 peft **0.18.1**（避免再升到 0.19 触发 MT-TE 问题）。

---

## 当前进度

| 阶段 | 状态 |
|------|------|
| pyproject 对齐 | 已完成（含变更 10 的 transformers pin） |
| peft / ray 安装 | 已完成；peft 已降至 **0.18.1** |
| MUSA Platform | 已完成 |
| server_config / run.sh（MUSA） | 已完成 |
| `pip install -e . --no-deps` | 已完成 |
| run.sh PID/线程加固 | 已完成（变更 7） |
| tinker 3.10 适配 | 已完成（变更 8） |
| self_cognition 本地数据/模型 | 已完成（变更 9） |
| transformers 5.2.0 | 已完成（变更 10） |
| 起 Server + 跑 `self_cognition.py` | **验证中 / 待重试** |

---

## 回滚

```bash
# pyproject
git checkout -- pyproject.toml
# 或备份文件（若有）

# Platform
rm -f src/twinkle/utils/platforms/musa.py
git checkout -- \
  src/twinkle/utils/platforms/base.py \
  src/twinkle/utils/platforms/__init__.py \
  src/twinkle/utils/__init__.py

# cookbook 恢复 CUDA 原版
cd cookbook/client/server/transformer
cp server_config.yaml.cuda.bak server_config.yaml
cp run.sh.cuda.bak run.sh

# editable twinkle
pip uninstall twinkle-kit

# peft 若需回到 0.19（一般不建议，除非 TE 已修好）
# pip install 'peft==0.19.0'

# transformers 回滚（若 5.2 有问题）
# pip install 'transformers==4.50.2' 'tokenizers==0.21.4' 'huggingface-hub==0.34.4'
```
