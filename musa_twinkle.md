# musa_twinkle

本文记录在 **Moore Threads MUSA** 环境下，为适配 Twinkle（跑通 `cookbook/client/twinkle/self_host/self_cognition.py`）所做的全部修改：依赖/`pyproject.toml`、pip 安装、**MUSA Platform 代码**、以及 **Server cookbook（server_config.yaml / run.sh）**。

> 原则：不覆盖、不重装现有 `torch` / `torch_musa`；依赖版本以本机实测为准；Platform 仿 NPU 最小实现。

---

## 1. 环境实测（修改前采集）

采集命令：

```bash
python -c "import sys; print(sys.version)"
python -c "import torch, torch_musa; print(torch.__version__, torch.musa.is_available())"
pip show torch torch_musa accelerate transformers modelscope fastapi omegaconf datasets safetensors numpy scipy | grep -E '^(Name|Version):'
```

| 组件 | 实测版本 | 备注 |
|------|----------|------|
| Python | 3.10.12 | 原 pyproject 要求 `>=3.11`，需放宽 |
| torch | 2.7.1 | MUSA 配套，**禁止被 pip 覆盖** |
| torch_musa | 2.7.1 | `torch.musa.is_available() == True` |
| torchvision | 0.22.1 | MUSA 配套（本地显示 `0.22.1+32091f2`） |
| accelerate | 1.12.0 | 已识别 MUSA |
| transformers | 4.50.2 | 已安装 |
| modelscope | 1.36.3 | 已安装 |
| fastapi | 0.116.1 | 已安装 |
| omegaconf | 2.3.0 | 已安装 |
| datasets | 4.4.2 | 已安装 |
| safetensors | 0.6.2 | 已安装 |
| numpy | 1.26.0 | 原要求 `>=2.0,<2.3`，**会冲突**，必须下调 |
| scipy | 1.10.1 | 原未声明；补钉死，避免连带升级 numpy |
| peft | 未安装 → 后已装 `0.19.0` | 见第 5、8 节 |
| ray | 未安装 → 后已装 `2.55.1` | 见第 5、8 节 |
| pydantic | 原 `2.11.7` → 后随 Ray 升至 `2.13.4` | 见第 6 节 |
| 硬件 | MTT S5000 | `musaInfo` 可见 |

---

## 2. `pyproject.toml` 修改前（仓库原版要点）

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

**与本机冲突点：**

1. Python 3.10 不满足 `>=3.11`
2. `numpy==1.26.0` 不满足 `>=2.0,<2.3`（若强行升级可能牵动 MUSA 栈）
3. `torch` / `torchvision` 无精确 pin，裸装 extras 可能拉到 CUDA 官方 wheel
4. `peft` / `ray` / `accelerate` 无精确目标版本，解析结果不确定

---

## 3. `pyproject.toml` 修改项一览（对齐本机）

| 字段 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| `requires-python` | `>=3.11,<3.13` | `>=3.10,<3.13` | 本机 Python 3.10.12 |
| `numpy` | `>=2.0.0,<2.3.0` | `==1.26.0` | 与现有环境一致，保护 MUSA |
| `scipy` | （无） | `==1.10.1` | 钉死，避免间接升级 numpy |
| `datasets` | 未 pin | `==4.4.2` | 对齐实测 |
| `omegaconf` | `>=2.3.0,<3.0.0` | `==2.3.0` | 对齐实测 |
| `fastapi` | 未 pin | `==0.116.1` | 对齐实测 |
| `modelscope` | `>=1.34.0` | `==1.36.3` | 对齐实测 |
| `safetensors` | 未 pin | `==0.6.2` | 对齐实测 |
| `transformers` | 未 pin | `==4.50.2` | 对齐实测 |
| `peft` | `>=0.11.0,<=0.19.0` | `==0.19.0` | 锁定安装目标 |
| `accelerate` | 未 pin | `==1.12.0` | 对齐实测 |
| `torch` | `>=2.6.0,<3.0.0` | `==2.7.1` | 对齐 MUSA 配套 |
| `torchvision` | 未 pin | `==0.22.1` | 对齐 MUSA 配套 |
| `ray` | `ray[serve]` | `ray[serve]==2.55.1` | 锁定安装目标 |

修改后文件路径：`/data/wangmx/twinkle/pyproject.toml`。

**说明：** `torch_musa`、`pydantic` 不写入主依赖 pin（前者由 MUSA 官方栈提供；后者为 Ray 传递依赖，装 Ray 后环境为 `2.13.4`）。

---

## 4. 修改前操作（建议先做）

```bash
# 4.1 确认 MUSA 可用，并记录版本
python -c "import torch, torch_musa; print(torch.__version__, torch_musa.__version__, torch.musa.is_available())"
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"

# 4.2 备份原 pyproject（可选）
cp /data/wangmx/twinkle/pyproject.toml /data/wangmx/twinkle/pyproject.toml.bak.pre_musa

# 4.3 查看原约束
sed -n '1,30p' /data/wangmx/twinkle/pyproject.toml
```

---

## 5. 已执行的 pip 安装（修改后操作）

### 5.1 兼容性分析结论（安装前）

| 包 | 会否动 `torch`/`torch_musa` | 结论 |
|----|------------------------------|------|
| `peft==0.19.0` | 否（dry-run 只装 peft） | 可直接装 |
| `ray[serve]==2.55.1` | 否（不依赖 torch/numpy） | 可直接装；会连带升级 pydantic |

### 5.2 实际执行命令

```bash
pip install 'peft==0.19.0' 'ray[serve]==2.55.1'
```

（未使用 `pip install -e '.[transformers,ray]'`，避免解析到 CUDA 官方 torch。）

### 5.3 安装结果

| 项 | 结果 |
|----|------|
| peft | 已安装 `0.19.0` |
| ray | 已安装 `2.55.1` |
| torch / torch_musa | **未变**：`2.7.1`，`torch.musa.is_available() == True` |
| numpy / scipy | **未变**：`1.26.0` / `1.10.1` |
| pydantic | **连带升级**：`2.11.7` → `2.13.4`（Ray Serve 排除 2.10/2.11） |

### 5.4 安装后自检命令

```bash
python -c "import torch, torch_musa, peft, ray, pydantic; print(torch.__version__, torch.musa.is_available(), peft.__version__, ray.__version__, pydantic.__version__)"
```

### 5.5 Twinkle 本体安装（如尚未执行）

```bash
cd /data/wangmx/twinkle
pip install -e . --no-deps
```

**不推荐：**

```bash
pip install -e '.[transformers,ray]'   # 可能覆盖 torch/torchvision
```

---

## 6. pydantic 升级说明（`2.11.7` → `2.13.4`）

- **原因：** 安装 `ray[serve]==2.55.1` 时，Ray 声明排除 `pydantic 2.10.*` / `2.11.*`，pip 解析到 `2.13.4`。
- **对 torch_musa：** 无影响（pydantic 与 torch 无关）。
- **对 Twinkle / FastAPI：** 仍在 Pydantic v2 内小版本升级；Twinkle 主要用 `BaseModel` / `Field` / `model_dump` 等常规 API，风险较低。
- **决策：** 已接受随 Ray 升级；无需单独先装 pydantic，也不必拆成两步。

---

## 7. 安装后与 `pyproject.toml` 对齐检查

检查结论：**pyproject 中已声明的 pin 与环境全部对齐。**

| pyproject 声明 | 环境实测 | 状态 |
|----------------|----------|------|
| `requires-python >=3.10,<3.13` | Python 3.10.12 | 对齐 |
| `numpy==1.26.0` | 1.26.0 | 对齐 |
| `scipy==1.10.1` | 1.10.1 | 对齐 |
| `datasets==4.4.2` | 4.4.2 | 对齐 |
| `omegaconf==2.3.0` | 2.3.0 | 对齐 |
| `fastapi==0.116.1` | 0.116.1 | 对齐 |
| `modelscope[framework]==1.36.3` | 1.36.3 | 对齐 |
| `safetensors==0.6.2` | 0.6.2 | 对齐 |
| `transformers==4.50.2` | 4.50.2 | 对齐 |
| `peft==0.19.0` | 0.19.0 | 对齐 |
| `accelerate==1.12.0` | 1.12.0 | 对齐 |
| `torch==2.7.1` | 2.7.1（musa 可用） | 对齐 |
| `torchvision==0.22.1` | `0.22.1+32091f2` | 对齐（本地构建后缀） |
| `ray[serve]==2.55.1` | 2.55.1 | 对齐 |

**文档层面备注（非版本冲突）：**

1. `pyproject.toml` 中 peft/ray 旁注释已改为「已安装」。
2. `pydantic==2.13.4`、`torch_musa` 未写入 pyproject（传递依赖 / 官方栈），属预期。

---

## 8. MUSA Platform 代码适配（已完成）

### 8.1 修改前现状

仓库原平台目录仅有：

| 文件 | 平台 |
|------|------|
| `src/twinkle/utils/platforms/gpu.py` | GPU / CUDA |
| `src/twinkle/utils/platforms/npu.py` | 昇腾 NPU |
| `src/twinkle/utils/platforms/mps.py` | Apple MPS |
| `src/twinkle/utils/platforms/base.py` | 注册入口 |

- `Platform.get_platform_names()` 原为 `['GPU', 'NPU', 'MPS']`，**无 MUSA**
- `Platform.get_platform('musa')` → `ValueError: Unsupported platform`
- 无 `nvidia-smi` / `npu-smi` 时自动探测会 **默认落到 GPU/`cuda`**，在本机（仅 MUSA）上会导致 `.to('cuda:0')` 失败
- 可参考模板：`npu.py`（`ASCEND_RT_VISIBLE_DEVICES` / `npu` / `hccl`）

本机实测支撑适配的关键信息：

| 项 | 值 |
|----|-----|
| 硬件 | 8× MTT S5000 |
| `torch.musa.is_available()` | True |
| 可见设备环境变量 | `MUSA_VISIBLE_DEVICES`（有效）；另有 `MTHREADS_VISIBLE_DEVICES` |
| 集合通信 | `torch.distributed.is_mccl_available() == True`，后端名 `mccl` |
| 探测命令 | `mthreads-gmi`、`musaInfo` 均存在 |

### 8.2 修改文件清单

| 操作 | 路径 | 说明 |
|------|------|------|
| **新增** | `src/twinkle/utils/platforms/musa.py` | MUSA Platform 实现（仿 NPU 精简版） |
| **修改** | `src/twinkle/utils/platforms/base.py` | 注册 MUSA；自动探测优先于 CUDA |
| **修改** | `src/twinkle/utils/platforms/__init__.py` | 导出 `MUSA` / `ensure_musa_backend` / `is_musa_available` |
| **修改** | `src/twinkle/utils/__init__.py` | 对外导出同上符号 |

### 8.3 新增 `musa.py` 详细设计

路径：`/data/wangmx/twinkle/src/twinkle/utils/platforms/musa.py`

#### 8.3.1 辅助函数

| 符号 | 作用 |
|------|------|
| `ensure_musa_backend()` | `import torch_musa`；失败则抛 `RuntimeError`，提示安装 torch_musa |
| `is_musa_available()` | `@lru_cache`；若存在 `mthreads-gmi` / `musaInfo`，或 `torch.musa.is_available()`，则返回 True |
| `_resolve_musa_physical_device_id(device_id)` | 按 `MUSA_VISIBLE_DEVICES` 把本地逻辑下标映射到物理卡号 |

#### 8.3.2 类 `MUSA(Platform)` API（对齐 NPU/GPU）

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `visible_device_env()` | `'MUSA_VISIBLE_DEVICES'` | 不用 `CUDA_VISIBLE_DEVICES` |
| `device_prefix()` | `'musa'` | 设备字符串前缀 |
| `get_local_device(idx)` | `f'musa:{idx}'` | 如 `musa:0` |
| `device_backend()` | `'mccl'` | 多卡集合通信；单卡 `ranks=1` 时不一定初始化 PG |
| `get_vllm_device_uuid(device_id)` | 设备名:物理 id，或 sha1 截断 | 无 vLLM-MUSA 时用确定性 fallback（仿 NPU） |

设计取舍：

- **不做** HCCL 那套端口环境推导（NPU 特有）；MUSA 用 MCCL，首期单卡训练不依赖复杂端口布局
- **不做** 对 `npu-smi` 式 Bus-Id 解析；UUID 优先 `torch.musa.get_device_name(physical_id)`
- 与 `gpu.py` 同级接口，保证 Ray `ResourceManager` / `ray_helper` 用 `Platform.__name__`（`MUSA`）挂自定义资源时能对上

### 8.4 `base.py` 注册改动（修改前后）

**修改前（自动探测）：**

```
npu-smi → NPU
nvidia-smi → GPU
MPS → MPS
否则 → GPU（默认）
```

**修改后（自动探测）：**

```
npu-smi → NPU
is_musa_available() → MUSA   # 新增；优先于 CUDA，避免本机误判为 cuda
nvidia-smi → GPU
MPS → MPS
否则 → GPU
```

**显式指定：**

- 新增：`platform.upper() in ('MUSA',)` → `MUSA` + `ensure_musa_backend()`
- `get_platform_names()`：`['GPU', 'NPU', 'MUSA', 'MPS']`

### 8.5 导出改动

`platforms/__init__.py`：

```python
from .musa import MUSA, ensure_musa_backend, is_musa_available
```

`utils/__init__.py`：

```python
from .platforms import (GPU, MUSA, NPU, Platform, ensure_hccl_socket_env,
                        ensure_musa_backend, ensure_npu_backend, is_musa_available)
```

### 8.6 验证结果（本机已跑通）

```bash
PYTHONPATH=/data/wangmx/twinkle/src python - <<'PY'
from twinkle.utils.platforms import Platform, MUSA, ensure_musa_backend, is_musa_available
print('is_musa_available', is_musa_available())
print('names', Platform.get_platform_names())
p = Platform.get_platform()
print('auto', p.__name__, p.device_prefix(), p.visible_device_env(), p.device_backend())
print('local', Platform.get_local_device())
print('explicit', Platform.get_platform('musa').get_local_device(0))
print('uuid', Platform.get_platform('musa').get_vllm_device_uuid(0))
ensure_musa_backend()
print('OK')
PY
```

实测输出：

```text
is_musa_available True
names ['GPU', 'NPU', 'MUSA', 'MPS']
auto MUSA musa MUSA_VISIBLE_DEVICES mccl
local musa:0
explicit musa:0
uuid MTT_S5000:0
OK
```

### 8.7 本步之后的 cookbook / 安装进展

Platform 完成后已继续完成 **server_config.yaml / run.sh**（见第 12 节）。  
`pip install -e . --no-deps` **尚未执行**（影响分析见第 13 节，由你决定是否执行）。

可选后补（单卡非必须）：`framework.py` / `grad_clip.py` 的 musa 分支；`resource_manager.noset_env` 增加 `RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES`（`run.sh` 已先 export 同名变量）。

---

## 9. 变更时间线（摘要）

1. 按本机实测修改 `pyproject.toml`（第 3 节）。
2. 分析 `peft` / `ray` 对 `torch_musa` 的影响 → 可直接 pip。
3. 分析 pydantic `2.11 → 2.13` → 可接受。
4. 执行：`pip install 'peft==0.19.0' 'ray[serve]==2.55.1'`。
5. 复查：pyproject pin 与环境对齐；`torch_musa` 正常。
6. 新增 MUSA Platform（`musa.py`）并注册；验证 `Platform.get_platform()` → `MUSA`。
7. **改写 cookbook `server_config.yaml` / `run.sh` 为 MUSA 单卡最小配置；CUDA 原文件备份为 `*.cuda.bak`。**
8. （待你决定）`pip install -e . --no-deps` → 起 Server → 跑 `self_cognition.py`。

---

## 10. 与跑通 `self_cognition.py` 的关系

| 阶段 | 内容 | 状态 |
|------|------|------|
| 依赖与 pyproject | 对齐本机、装 peft/ray | **已完成** |
| MUSA Platform | `musa.py` + 注册 | **已完成** |
| Server cookbook | yaml `device_type: musa`、`run.sh` Ray 资源 | **已完成**（见第 12 节） |
| 安装 twinkle | `pip install -e . --no-deps` | **未执行**（分析见第 13 节） |
| 起服 + 客户端 | Server :8000 → `self_cognition.py` | **未做** |

脚本本身（`self_cognition.py`）一般无需改逻辑；它是 HTTP 客户端。

---

## 11. cookbook：`server_config.yaml` / `run.sh`（已改写）

目录：`/data/wangmx/twinkle/cookbook/client/server/transformer/`

| 文件 | 操作 |
|------|------|
| `server_config.yaml` | 已改为 MUSA 配置 |
| `run.sh` | 已改为 MUSA + Ray 自定义资源启动 |
| `server_config.yaml.cuda.bak` | CUDA 原版备份 |
| `run.sh.cuda.bak` | CUDA 原版备份 |

### 11.1 `server_config.yaml` 修改要点

| 项 | 修改前（CUDA） | 修改后（MUSA） |
|----|----------------|----------------|
| model `device_group.device_type` | `cuda` | `musa` |
| model `device_mesh.device_type` | `cuda` | `musa` |
| model `ranks` / `nproc_per_node` / `dp_size` | 1 | 1（单卡首跑） |
| model `model_id` | `ms://Qwen/Qwen3.5-4B` | `/data/wangmx/Qwen3.5-4B`（本地权重，避免再下） |
| `use_megatron` | `false` | `false`（不变） |
| sampler 整段 | 启用（`device_type: cuda`） | **整段注释掉**（`self_cognition` 不 sample） |
| processor | `CPU` | `CPU`（不变） |
| `supported_models` / route | `Qwen/Qwen3.5-4B` | 不变（需与客户端 `base_model` 一致） |

### 11.2 `run.sh` 修改要点

| 项 | 修改前（CUDA） | 修改后（MUSA） |
|----|----------------|----------------|
| 可见设备 | `CUDA_VISIBLE_DEVICES=0,1,2,3` 等 | `MUSA_VISIBLE_DEVICES=0`（单卡） |
| Ray GPU | `--num-gpus=4` | `--num-gpus=0` |
| 加速卡资源 | 依赖 Ray GPU | `--resources='{"MUSA":1}'`（与 `Platform.__name__` 一致） |
| 节点布局 | head 4 卡 + worker 4 卡 + CPU worker | head 1×MUSA + CPU worker |
| 其它 | 无 | `ray stop --force` 后启动；export `RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES=1` |

启动命令（你决定执行时）：

```bash
cd /data/wangmx/twinkle/cookbook/client/server/transformer
bash run.sh
```

多卡示例已写在 `run.sh` 注释中；启用时需同步加大 yaml 里的 `ranks` / `nproc_per_node` / `MUSA` 数量。

---
