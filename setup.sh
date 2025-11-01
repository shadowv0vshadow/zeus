#!/bin/bash
# YouDub 项目设置脚本
# 用于 Mac 和 Linux 系统

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}YouDub 项目设置脚本${NC}"
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

# 安装 PyTorch（demucs 要求 torchaudio<2.1，所以使用特定版本）
echo -e "\n${YELLOW}安装 PyTorch（兼容 demucs 版本）...${NC}"
# 检测是否有 NVIDIA GPU 且 CUDA 可用
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    # 进一步检查 CUDA 是否真的可用（nvidia-smi 存在不代表 PyTorch 能用 CUDA）
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
    fi
fi

# 支持环境变量强制 CPU 模式
FORCE_CPU="${FORCE_CPU:-false}"
if [[ "$FORCE_CPU" == "true" ]] || [[ "$FORCE_CPU" == "1" ]]; then
    HAS_GPU=false
    echo -e "${YELLOW}环境变量 FORCE_CPU=true，强制使用 CPU 模式${NC}"
fi

# 检查是否需要安装或重新安装（确保版本兼容 demucs）
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
TORCHAUDIO_VERSION=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "")

if [[ -z "$TORCH_VERSION" ]] || [[ "$TORCHAUDIO_VERSION" != "2.0.2" ]] || [[ "${TORCHAUDIO_VERSION:0:3}" > "2.0" ]]; then
    echo "安装兼容 demucs 的 PyTorch 版本（torch 2.0.1, torchaudio 2.0.2）..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    
    if [[ "$HAS_GPU" == true ]]; then
        echo "检测到 NVIDIA GPU，安装 CUDA 版本"
        pip install torch==2.0.1 torchvision torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 || \
        pip install torch==2.0.1 torchvision torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121
    else
        echo "未检测到 NVIDIA GPU，安装 CPU 版本"
        pip install torch==2.0.1 torchvision torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
    fi
    
    echo -e "${GREEN}PyTorch 安装完成${NC}"
else
    echo "PyTorch 版本已兼容: torch $TORCH_VERSION, torchaudio $TORCHAUDIO_VERSION"
fi

# 安装其他依赖（先安装，因为 demucs 可能需要一些基础依赖）
echo -e "\n${YELLOW}安装依赖包...${NC}"
# 创建临时 requirements 文件，排除 demucs（单独安装）
grep -v "demucs" requirements.txt > /tmp/requirements_no_demucs.txt || true
if [ -s /tmp/requirements_no_demucs.txt ]; then
    pip install -r /tmp/requirements_no_demucs.txt
    rm /tmp/requirements_no_demucs.txt
else
    echo "警告: 无法创建临时 requirements 文件"
fi

# 安装 demucs（从 git，必须在 PyTorch 之后安装）
echo -e "\n${YELLOW}安装 demucs...${NC}"
if ! pip show demucs &> /dev/null; then
    echo "安装 demucs 的依赖..."
    pip install dora-search einops "julius>=0.2.3" "lameenc>=1.2" openunmix pyyaml tqdm
    
    echo "安装 demucs（从 GitHub）..."
    pip install git+https://github.com/facebookresearch/demucs#egg=demucs
    
    # 验证安装
    if python3 -c "import demucs" 2>/dev/null; then
        echo -e "${GREEN}demucs 安装成功${NC}"
    else
        echo -e "${YELLOW}警告: demucs 导入测试失败，将在依赖修复中处理${NC}"
    fi
else
    echo "demucs 已安装"
fi

# 运行依赖修复（自动检测并修复兼容性问题）
echo -e "\n${GREEN}================================"
echo -e "开始依赖兼容性修复${NC}"
echo -e "${GREEN}================================${NC}"

# 1. 修复 NumPy（如果 pyannote.audio 使用 np.NaN）
echo -e "\n${YELLOW}[1/7] 检查 NumPy 版本...${NC}"
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "未安装")
if python3 -c "import numpy; hasattr(numpy, 'NaN')" 2>/dev/null; then
    echo "NumPy 版本: $NUMPY_VERSION（兼容）"
elif [[ "$NUMPY_VERSION" == "2."* ]]; then
    echo "检测到 NumPy 2.0+，尝试测试 pyannote.audio 兼容性..."
    if ! python3 -c "from pyannote.audio import Model, Inference" 2>/dev/null; then
        echo "降级 NumPy 到 1.x 版本..."
        pip uninstall -y numpy 2>/dev/null || true
        pip install "numpy==1.26.4" || pip install "numpy<2.0,>=1.21.0"
    else
        echo "NumPy 2.0 可用，保持当前版本"
    fi
fi

# 2. 修复 websockets 和 gradio-client（gradio 兼容性）
echo -e "\n${YELLOW}[2/7] 检查 websockets 和 gradio-client 版本...${NC}"
WS_VERSION=$(pip show websockets 2>/dev/null | grep Version | cut -d' ' -f2 || echo "未安装")
GRADIO_CLIENT_VERSION=$(pip show gradio-client 2>/dev/null | grep Version | cut -d' ' -f2 || echo "未安装")

# 先检查并降级 gradio-client（如果 >= 1.13.0）
if [[ "$GRADIO_CLIENT_VERSION" != "未安装" ]]; then
    MAJOR=$(echo "$GRADIO_CLIENT_VERSION" | cut -d'.' -f1)
    MINOR=$(echo "$GRADIO_CLIENT_VERSION" | cut -d'.' -f2)
    if [[ "$MAJOR" -eq 1 ]] && [[ "$MINOR" -ge 13 ]]; then
        echo "检测到 gradio-client $GRADIO_CLIENT_VERSION，降级到兼容版本..."
        pip uninstall -y gradio-client 2>/dev/null || true
        pip install "gradio-client==1.12.0" || pip install "gradio-client<1.13.0,>=1.10.0"
    fi
fi

# 测试 websockets.asyncio
if ! python3 -c "from websockets.asyncio.client import ClientConnection" 2>/dev/null; then
    echo "websockets.asyncio 不可用，安装兼容版本..."
    pip uninstall -y websockets 2>/dev/null || true
    pip install "websockets==12.0" || pip install "websockets>=11.0,<13.0"
fi

# 3. 修复 PyTorch 和 torchaudio（确保版本正确）
echo -e "\n${YELLOW}[3/7] 检查 PyTorch 和 torchaudio 版本...${NC}"
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装")
TORCHAUDIO_VERSION=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "未安装")

# 检查 CUDA 库加载问题
TORCHAUDIO_ERROR=$(python3 -c "import torchaudio" 2>&1 || echo "")
if echo "$TORCHAUDIO_ERROR" | grep -q "libcudart\|OSError.*Could not load\|cannot open shared object file"; then
    echo "检测到 CUDA 库问题，安装 CPU 版本..."
    pip uninstall -y torchaudio torch torchvision 2>/dev/null || true
    pip install "torch==2.0.1" "torchvision" "torchaudio==2.0.2" --index-url https://download.pytorch.org/whl/cpu
elif [[ "$TORCH_VERSION" == "1."* ]] || [[ "$TORCHAUDIO_VERSION" != "2.0.2" ]]; then
    echo "检测到版本不匹配，重新安装兼容版本..."
    pip uninstall -y torchaudio torch torchvision 2>/dev/null || true
    if command -v nvidia-smi &> /dev/null; then
        pip install "torch==2.0.1" "torchvision" "torchaudio==2.0.2" --index-url https://download.pytorch.org/whl/cu118 || \
        pip install "torch==2.0.1" "torchvision" "torchaudio==2.0.2" --index-url https://download.pytorch.org/whl/cu121
    else
        pip install "torch==2.0.1" "torchvision" "torchaudio==2.0.2" --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# 4. 修复 pytorch-lightning（pyannote.audio 需要）
echo -e "\n${YELLOW}[4/7] 检查 pytorch-lightning 版本...${NC}"
PL_VERSION=$(pip show pytorch-lightning 2>/dev/null | grep Version | cut -d' ' -f2 || echo "未安装")

# 检查 cloud_io 模块
if ! python3 -c "from pytorch_lightning.utilities.cloud_io import load" 2>/dev/null; then
    echo "cloud_io 不可用，重新安装 pytorch-lightning 2.0.9..."
    pip uninstall -y pytorch-lightning lightning lightning-fabric 2>/dev/null || true
    pip install "pytorch-lightning==2.0.9" || pip install "pytorch-lightning==2.0.0"
    pip install "lightning==2.0.9" || pip install "lightning==2.0.0"
elif [[ "$PL_VERSION" == "1."* ]]; then
    echo "检测到 pytorch-lightning 1.x，升级到 2.0.9..."
    pip uninstall -y pytorch-lightning lightning lightning-fabric 2>/dev/null || true
    pip install "pytorch-lightning==2.0.9"
    pip install "lightning==2.0.9"
elif ! python3 -c "import lightning_fabric" 2>/dev/null; then
    echo "lightning_fabric 缺失，安装 lightning 包..."
    pip install "lightning==2.0.9" || pip install "lightning==2.0.0"
fi

# 5. 修复 pyannote.audio（兼容性测试）
echo -e "\n${YELLOW}[5/7] 检查 pyannote.audio 兼容性...${NC}"
PYANNOTE_VERSION=$(pip show pyannote.audio 2>/dev/null | grep Version | cut -d' ' -f2 || echo "未安装")

if ! python3 -c "from pyannote.audio import Model, Inference" 2>/dev/null; then
    echo "pyannote.audio 导入失败，分析错误..."
    ERROR_MSG=$(python3 -c "from pyannote.audio import Model, Inference" 2>&1 || true)
    
    if echo "$ERROR_MSG" | grep -q "set_audio_backend"; then
        echo "检测到 set_audio_backend 问题，降级到 pyannote.audio 2.0.1..."
        pip uninstall -y pyannote.audio pyannote.core pyannote.database pyannote.metrics pyannote.pipeline 2>/dev/null || true
        pip install "pyannote.audio==2.0.1" || pip install "pyannote.audio==2.0.0"
    elif echo "$ERROR_MSG" | grep -q "pytorch_lightning.utilities.cloud_io"; then
        echo "检测到 cloud_io 问题，降级到 pyannote.audio 2.0.1..."
        pip uninstall -y pyannote.audio pyannote.core pyannote.database pyannote.metrics pyannote.pipeline 2>/dev/null || true
        pip install "pyannote.audio==2.0.1" || pip install "pyannote.audio==2.0.0"
    elif echo "$ERROR_MSG" | grep -q "np.NaN"; then
        echo "NumPy 兼容性问题，降级 NumPy..."
        pip uninstall -y numpy 2>/dev/null || true
        pip install "numpy==1.26.4"
    else
        echo "尝试降级到 pyannote.audio 2.0.1..."
        pip uninstall -y pyannote.audio pyannote.core pyannote.database pyannote.metrics pyannote.pipeline 2>/dev/null || true
        pip install "pyannote.audio==2.0.1"
    fi
fi

# 6. 修复其他依赖冲突
echo -e "\n${YELLOW}[6/7] 修复其他依赖冲突...${NC}"
HF_VERSION=$(pip show huggingface-hub 2>/dev/null | grep Version | cut -d' ' -f2 || echo "未安装")
if [[ "$HF_VERSION" == "0.8."* ]] || [[ "$HF_VERSION" == "未安装" ]]; then
    echo "升级 huggingface-hub..."
    pip install "huggingface-hub>=0.33.5,<2.0" || pip install "huggingface-hub>=0.19.3,<2.0"
fi

SF_VERSION=$(pip show soundfile 2>/dev/null | grep Version | cut -d' ' -f2 || echo "未安装")
if [[ "$SF_VERSION" == "0.10."* ]]; then
    echo "升级 soundfile..."
    pip install "soundfile>=0.12.1"
fi

# 修复 pandas 版本（gradio 5.49.1 需要兼容版本）
PANDAS_VERSION=$(pip show pandas 2>/dev/null | grep Version | cut -d' ' -f2 || echo "未安装")
if [[ "$PANDAS_VERSION" == "2.2."* ]] || [[ "$PANDAS_VERSION" == "2.3."* ]]; then
    echo "检测到 pandas 2.2.x/2.3.x，降级到兼容版本（避免 infer_objects copy 参数问题）..."
    pip uninstall -y pandas 2>/dev/null || true
    pip install "pandas>=2.0.0,<2.2.0" || pip install "pandas>=1.5.0,<2.2.0"
fi

# 7. 验证所有依赖
echo -e "\n${YELLOW}[7/7] 验证所有依赖...${NC}"
python3 -c "import numpy; print(f'✓ numpy: {numpy.__version__}')" || echo "✗ numpy"
python3 -c "import torch; print(f'✓ torch: {torch.__version__}')" || echo "✗ torch"
python3 -c "import torchaudio; print(f'✓ torchaudio: {torchaudio.__version__}')" || echo "✗ torchaudio"
python3 -c "from websockets.asyncio.client import ClientConnection; print('✓ websockets.asyncio')" 2>/dev/null || echo "✗ websockets.asyncio"
python3 -c "import gradio; print('✓ gradio')" 2>/dev/null || echo "✗ gradio"
python3 -c "from pytorch_lightning.utilities.cloud_io import load; print('✓ cloud_io')" 2>/dev/null || echo "✗ cloud_io"
python3 -c "from pyannote.audio import Model, Inference; print('✓ pyannote.audio')" 2>/dev/null || echo "✗ pyannote.audio"
if pip show demucs &> /dev/null; then
    python3 -c "import demucs; print('✓ demucs')" 2>/dev/null || echo "✗ demucs"
fi

# 安装 TTS（如果需要）
echo -e "\n${YELLOW}安装 TTS...${NC}"
pip install TTS || echo -e "${YELLOW}警告: TTS 安装失败或已安装${NC}"

echo -e "\n${GREEN}================================"
echo -e "设置完成！${NC}"
echo -e "${YELLOW}检查依赖冲突:${NC}"
pip check 2>&1 | head -10 || echo "没有发现严重依赖冲突"
echo -e "\n${YELLOW}要激活虚拟环境，请运行:${NC}"
echo -e "  source venv/bin/activate"
echo -e "${YELLOW}要启动应用，请运行:${NC}"
echo -e "  ./run.sh${NC}"

