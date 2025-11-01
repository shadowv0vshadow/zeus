# YouDub CPU版本安装指南

本指南适用于在**无GPU的Linux服务器**上运行YouDub。

## 快速安装（CPU版本）

### 方式一：使用专用CPU安装脚本（推荐）

```bash
# 运行CPU专用安装脚本
./setup-cpu.sh
```

### 方式二：使用环境变量强制CPU模式

```bash
# 设置环境变量强制CPU模式
export FORCE_CPU=true

# 运行标准安装脚本（会自动检测并安装CPU版本）
./setup.sh
```

### 方式三：手动安装

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 升级pip
pip install --upgrade pip

# 3. 安装PyTorch CPU版本（必须首先安装）
pip install torch==2.0.1 torchvision torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}')"
# 应该输出: PyTorch: 2.0.1+cpu, CUDA可用: False
```

## 关键配置

### 1. PyTorch CPU版本

确保安装的是CPU版本，不是CUDA版本：

```bash
# 正确的CPU版本安装命令
pip install torch==2.0.1 torchvision torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# 验证
python -c "import torch; print(torch.__version__)"  # 应该包含 +cpu
python -c "import torch; print(torch.cuda.is_available())"  # 应该输出 False
```

### 2. 设备选择

代码中所有地方都支持 `device='auto'`，会自动检测GPU，如果没有GPU则使用CPU。

在UI中：
- Demucs Device: 选择 `'auto'` 或 `'cpu'`
- Whisper Device: 选择 `'auto'` 或 `'cpu'`

### 3. 性能优化建议

CPU模式下运行较慢，建议：

1. **使用较小的模型**：
   - Whisper: 使用 `small` 或 `medium` 而不是 `large`
   - Demucs: 使用 `htdemucs` 而不是 `htdemucs_ft`

2. **调整批处理大小**：
   - Whisper Batch Size: 降低到 16 或 8（默认32）

3. **考虑使用云服务**：
   - 字节跳动 TTS（不需要本地GPU）
   - 阿里云 TTS（不需要本地GPU）

## 常见问题

### Q: 安装后显示 torch: 2.9.0+cu128 而不是 CPU 版本

A: 说明安装了CUDA版本的PyTorch，需要卸载并重新安装CPU版本：
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### Q: websockets.asyncio 导入失败

A: **重要**：如果 `gradio` 可以正常导入和运行，`websockets.asyncio` 的导入错误通常可以忽略。

如果 `gradio` 无法导入，需要修复 websockets：
```bash
# 完全卸载并清理缓存
pip uninstall -y websockets gradio-client
pip cache purge
pip install --no-cache-dir "gradio-client==1.12.0"
pip install --no-cache-dir "websockets==12.0"

# 验证 gradio（这是最重要的）
python3 -c "import gradio as gr; print('✓ gradio 可以正常导入')"

# 或者运行修复脚本
./fix-websockets.sh
```

**注意**：`websockets 12.0` 可能没有 `websockets.asyncio` 子模块，但如果 `gradio` 能正常工作，说明它使用了其他 API 路径。关键是验证 `gradio` 能否正常导入和运行。

### Q: pyannote.audio 导入失败

A: 需要安装兼容版本的依赖：
```bash
pip install "einops>=0.3,<0.4.0"
pip install "soundfile>=0.10.2,<0.11"
pip install "huggingface-hub>=0.7,<0.9"
pip uninstall -y pyannote.audio pyannote.core pyannote.database pyannote.metrics pyannote.pipeline
pip install "pyannote.audio==2.0.1"
```

### Q: 运行时报错 "libcudart.so not found"

A: 说明安装了CUDA版本的PyTorch。按上面的步骤重新安装CPU版本。

### Q: NumPy 版本冲突（audiostretchy 需要 >=1.23.0，但安装了 1.22.0）

A: 升级 NumPy：
```bash
pip uninstall -y numpy
pip install "numpy>=1.23.0,<2.0"
```

### Q: gradio-client 版本冲突

A: `gradio-client 1.13.3` 需要 `websockets>=13.0`，但可能与 `gradio 5.49.1` 不兼容。

**解决方案**：降级 `gradio-client` 到兼容版本：
```bash
pip uninstall -y gradio-client websockets
pip install "gradio-client==1.12.0"
pip install "websockets==12.0"

# 验证
python3 -c "import gradio as gr; print('✓ gradio 可以正常导入')"
```

或者如果必须使用 `gradio-client 1.13.3`，尝试：
```bash
pip install "websockets>=13.0,<16.0"
# 注意：websockets 13.0+ 可能改变了 API，gradio 可能需要适配
```

### Q: ImportError: cannot import name 'SpaceHardware' from 'huggingface_hub'

A: 这是 `huggingface-hub` 版本冲突问题。`gradio-client` 需要新版本，但 `pyannote.audio` 需要旧版本。

**解决方案（推荐方案1）**：
```bash
# 如果不使用字节跳动 TTS，可以移除 pyannote.audio
pip uninstall -y pyannote.audio pyannote.core pyannote.database pyannote.metrics pyannote.pipeline
pip install "huggingface-hub>=0.19.3,<2.0"

# 或者运行修复脚本
./fix-huggingface-hub.sh
```

**或者尝试兼容版本**：
```bash
pip install "huggingface-hub==0.19.4"
```

### Q: CPU模式下运行太慢

A: 这是正常的。CPU模式比GPU慢10-100倍。建议：
- 使用更小的模型（Whisper small/medium）
- 考虑在云服务器上运行（如带GPU的实例）
- 或使用API服务（字节跳动、阿里云TTS）

## 验证安装

运行以下命令验证所有组件：

```bash
source venv/bin/activate

# 检查PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 检查关键模块
python -c "import demucs; print('✓ demucs')"
python -c "import whisperx; print('✓ whisperx')"
python -c "from pyannote.audio import Model; print('✓ pyannote.audio')"
python -c "import gradio; print('✓ gradio')"
```

## 启动应用

```bash
source venv/bin/activate
python app.py
```

或使用启动脚本：

```bash
./run.sh
```

## 依赖版本说明

CPU版本与GPU版本使用相同的依赖版本，只是PyTorch安装源不同：

- **GPU版本**: `--index-url https://download.pytorch.org/whl/cu118`
- **CPU版本**: `--index-url https://download.pytorch.org/whl/cpu`

所有其他依赖（demucs, whisperx, pyannote.audio等）都是相同的。

