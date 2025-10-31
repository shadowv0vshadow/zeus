#!/bin/bash
# 激活虚拟环境的简单脚本
# 用于 Mac 和 Linux 系统

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "错误: 虚拟环境不存在"
    echo "请先运行设置脚本:"
    echo "  ./setup.sh"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

echo "虚拟环境已激活"
echo "提示: 要退出虚拟环境，请运行 'deactivate'"

# 如果是在交互式 shell 中运行，保持激活状态
if [ -t 0 ]; then
    exec $SHELL
fi

