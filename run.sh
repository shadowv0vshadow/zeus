#!/bin/bash
# YouDub 应用启动脚本
# 用于 Mac 和 Linux 系统

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo -e "${RED}错误: 虚拟环境不存在${NC}"
    echo -e "${YELLOW}请先运行设置脚本:${NC}"
    echo "  ./setup.sh"
    exit 1
fi

# 激活虚拟环境
echo -e "${GREEN}激活虚拟环境...${NC}"
source venv/bin/activate

# 启动应用
echo -e "${GREEN}启动 YouDub 应用...${NC}"
python app.py

