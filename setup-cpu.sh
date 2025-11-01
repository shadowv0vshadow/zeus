#!/bin/bash
# YouDub CPU版本设置脚本（适用于无GPU的Linux服务器）

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}YouDub CPU版本项目设置脚本${NC}"
echo "================================"
echo -e "${YELLOW}注意: 此脚本配置为CPU模式，所有模型将在CPU上运行${NC}"
echo "================================"

# 检查 Python 3.10 或更高版本
echo -e "\n${YELLOW}检查 Python 版本...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: Python 3 未安装${NC}"
    echo "请从 https://www.python.org/downloads/ 下载并安装 Python 3.10 或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "检测到 Python 版本: $(python3 --version)"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}创建虚拟环境...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}虚拟环境创建成功${NC}"
else
    echo -e "\n${GREEN}虚拟环境已存在${NC}"
fi

# 激活虚拟环境
echo -e "\n${YELLOW}激活虚拟环境...${NC}"
source venv/bin/activate

# 升级 pip
echo -e "\n${YELLOW}升级 pip...${NC}"
python -m pip install --upgrade pip

# 强制安装 PyTorch CPU 版本
echo -e "\n${YELLOW}安装 PyTorch CPU 版本（兼容 demucs）...${NC}"
echo "卸载现有的 PyTorch 相关包..."
pip uninstall -y torch torchvision torchaudio pytorch 2>/dev/null || true

echo "安装 PyTorch 2.0.1 CPU 版本..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# 验证 PyTorch CPU 安装
echo -e "\n${YELLOW}验证 PyTorch 安装...${NC}"
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
TORCHAUDIO_VERSION=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "")
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [[ -z "$TORCH_VERSION" ]]; then
    echo -e "${RED}PyTorch 安装失败${NC}"
    exit 1
fi

echo "PyTorch版本: $TORCH_VERSION"
echo "torchaudio版本: $TORCHAUDIO_VERSION"
echo "CUDA可用: $CUDA_AVAILABLE"

# 检查是否正确安装了CPU版本
if echo "$TORCH_VERSION" | grep -q "cu"; then
    echo -e "${RED}错误: 检测到CUDA版本的PyTorch，但应该安装CPU版本${NC}"
    echo "重新安装CPU版本..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
fi

# 安装 NumPy（首先安装，因为很多包依赖它）
echo -e "\n${YELLOW}安装 NumPy（兼容版本）...${NC}"
pip uninstall -y numpy 2>/dev/null || true
# audiostretchy 需要 >=1.23.0，但 whisperx 需要 <2.1.0,>=2.0.2，我们选择 1.26.4（兼容两者）
pip install "numpy>=1.23.0,<2.0"

# 安装基础依赖
echo -e "\n${YELLOW}安装基础依赖包...${NC}"
pip install gradio==5.49.1
pip install loguru==0.7.3
pip install yt-dlp==2025.10.22
pip install scipy==1.15.3
pip install librosa==0.10.0
pip install python-dotenv==1.1.1
pip install openai==2.6.1
pip install dashscope==1.24.9
pip install audiostretchy==1.3.5
pip install bilibili-toolman==1.0.7.9

# 安装 websockets（必须在 gradio-client 之前）
echo -e "\n${YELLOW}安装 websockets...${NC}"
pip uninstall -y websockets 2>/dev/null || true
pip install "websockets>=11.0,<13.0"

# 安装 gradio-client（需要 websockets 11-13）
echo -e "\n${YELLOW}安装 gradio-client...${NC}"
pip uninstall -y gradio-client 2>/dev/null || true
pip install "gradio-client==1.13.3"

# 再次安装 websockets（确保兼容）
pip install "websockets==12.0"

# 安装 demucs 的依赖
echo -e "\n${YELLOW}安装 demucs 依赖...${NC}"
pip install dora-search einops "julius>=0.2.3" "lameenc>=1.2" openunmix pyyaml tqdm

# 安装 demucs（从 git）
echo -e "\n${YELLOW}安装 demucs...${NC}"
if ! pip show demucs &> /dev/null; then
    pip install git+https://github.com/facebookresearch/demucs#egg=demucs
    echo -e "${GREEN}demucs 安装成功${NC}"
else
    echo "demucs 已安装"
fi

# 安装 WhisperX（从 git）
echo -e "\n${YELLOW}安装 WhisperX...${NC}"
if ! pip show whisperx &> /dev/null; then
    pip install git+https://github.com/m-bain/whisperx.git
    echo -e "${GREEN}WhisperX 安装成功${NC}"
else
    echo "WhisperX 已安装"
fi

# 修复 pytorch-lightning（pyannote.audio 需要）
echo -e "\n${YELLOW}安装 pytorch-lightning...${NC}"
pip uninstall -y pytorch-lightning lightning lightning-fabric 2>/dev/null || true
pip install "pytorch-lightning==2.0.9"
pip install "lightning==2.0.9"

# 修复 pyannote.audio 及相关依赖
echo -e "\n${YELLOW}安装 pyannote.audio 及其兼容依赖...${NC}"
pip uninstall -y pyannote.audio pyannote.core pyannote.database pyannote.metrics pyannote.pipeline 2>/dev/null || true

# pyannote.audio 2.0.1 需要特定版本的依赖
pip install "einops>=0.3,<0.4.0"
pip install "soundfile>=0.10.2,<0.11"
pip install "huggingface-hub>=0.7,<0.9"

# 安装 pyannote.audio
pip install "pyannote.audio==2.0.1"

# whisperx 需要的依赖（但要注意 gradio 兼容性）
echo -e "\n${YELLOW}安装 whisperx 依赖...${NC}"
# whisperx 需要 pandas>=2.2.3，但 gradio 5.49.1 与 pandas 2.2.x 的 infer_objects(copy=True) 不兼容
# 使用 2.1.x 版本作为折中方案
pip install "pandas>=2.1.0,<2.2.0" || pip install "pandas>=2.0.0,<2.2.0"

# 可选：安装 TTS（如果需要 XTTS）
echo -e "\n${YELLOW}是否安装 TTS（XTTS模型）? [y/N]${NC}"
read -t 10 -n 1 INSTALL_TTS || INSTALL_TTS="n"
if [[ "$INSTALL_TTS" == "y" ]] || [[ "$INSTALL_TTS" == "Y" ]]; then
    echo "安装 TTS..."
    pip install TTS || echo -e "${YELLOW}警告: TTS 安装失败${NC}"
else
    echo "跳过 TTS 安装（可以使用字节跳动或阿里云 TTS）"
fi

# 验证所有依赖
echo -e "\n${YELLOW}验证所有依赖...${NC}"
echo "检查关键模块:"

TORCH_INFO=$(python3 -c "import torch; print(f'{torch.__version__}')" 2>/dev/null || echo "")
if [[ -n "$TORCH_INFO" ]]; then
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if echo "$TORCH_INFO" | grep -q "cu"; then
        echo -e "${RED}✗ torch: $TORCH_INFO (检测到CUDA版本，应该是CPU版本)${NC}"
    else
        echo -e "${GREEN}✓ torch: $TORCH_INFO (CPU模式)${NC}"
    fi
else
    echo "✗ torch"
fi

python3 -c "import torchaudio; print(f'✓ torchaudio: {torchaudio.__version__}')" 2>/dev/null || echo "✗ torchaudio"
python3 -c "import numpy; print(f'✓ numpy: {numpy.__version__}')" 2>/dev/null || echo "✗ numpy"

# 检查 websockets
if python3 -c "from websockets.asyncio.client import ClientConnection" 2>/dev/null; then
    echo "✓ websockets.asyncio"
else
    echo -e "${YELLOW}✗ websockets.asyncio (尝试修复...)${NC}"
    pip uninstall -y websockets 2>/dev/null || true
    pip install "websockets==12.0"
    python3 -c "from websockets.asyncio.client import ClientConnection; print('✓ websockets.asyncio (已修复)')" 2>/dev/null || echo "✗ websockets.asyncio (修复失败)"
fi

python3 -c "import gradio; print('✓ gradio')" 2>/dev/null || echo "✗ gradio"
python3 -c "from pytorch_lightning.utilities.cloud_io import load; print('✓ pytorch-lightning cloud_io')" 2>/dev/null || echo "✗ cloud_io"

# 检查 pyannote.audio
if python3 -c "from pyannote.audio import Model, Inference" 2>/dev/null; then
    echo "✓ pyannote.audio"
else
    echo -e "${YELLOW}✗ pyannote.audio (检查错误...)${NC}"
    ERROR_MSG=$(python3 -c "from pyannote.audio import Model, Inference" 2>&1 || true)
    echo "$ERROR_MSG" | head -3
fi

python3 -c "import demucs; print('✓ demucs')" 2>/dev/null || echo "✗ demucs"
python3 -c "import whisperx; print('✓ whisperx')" 2>/dev/null || echo "✗ whisperx"

# 验证设备选择
echo -e "\n${YELLOW}验证设备自动选择...${NC}"
python3 -c "import torch; device = 'cuda' if torch.cuda.is_available() else 'cpu'; print(f'自动设备选择: {device}')"

echo -e "\n${GREEN}================================"
echo -e "CPU版本设置完成！${NC}"
echo -e "${YELLOW}重要提示:${NC}"
echo -e "  - 所有模型将在CPU上运行，速度会比GPU慢"
echo -e "  - 建议使用较小的模型（如 whisper small/medium）"
echo -e "  - 在UI中选择 device='cpu' 或 'auto'（会自动选择CPU）"
echo -e "${YELLOW}检查依赖冲突:${NC}"
pip check 2>&1 | head -10 || echo "没有发现严重依赖冲突"
echo -e "\n${YELLOW}要激活虚拟环境，请运行:${NC}"
echo -e "  source venv/bin/activate"
echo -e "${YELLOW}要启动应用，请运行:${NC}"
echo -e "  ./run.sh${NC}"

