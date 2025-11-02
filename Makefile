.PHONY: format lint check install-hooks help

# 默认目标
help:
	@echo "可用命令:"
	@echo "  make format        - 格式化所有 Python 代码（Black）"
	@echo "  make lint          - 检查代码风格和错误（Ruff）"
	@echo "  make check         - 同时运行格式化和检查"
	@echo "  make fix           - 自动修复可以修复的问题（Ruff）"
	@echo "  make install-hooks - 安装 pre-commit hooks（首次运行）"
	@echo ""
	@echo "工具说明:"
	@echo "  - Black: 代码格式化工具"
	@echo "  - Ruff: 超快的 linter 和代码检查工具（替代 flake8, isort, pylint）"
	@echo "  - Pre-commit: Git hooks，提交前自动检查和格式化"

# 格式化代码
format:
	@echo "正在使用 Black 格式化代码..."
	black --line-length=120 youdub app.py

# 检查代码（Ruff）
lint:
	@echo "正在使用 Ruff 检查代码..."
	ruff check youdub app.py

# 自动修复可以修复的问题
fix:
	@echo "正在使用 Ruff 自动修复代码..."
	ruff check --fix youdub app.py
	@echo "正在使用 Black 格式化代码..."
	black --line-length=120 youdub app.py

# 完整检查（lint + format check）
check: lint format-check

# 检查格式化（不实际修改文件）
format-check:
	@echo "正在检查代码格式（不会修改文件）..."
	black --check --line-length=120 youdub app.py

# 安装 pre-commit hooks
install-hooks:
	@echo "正在安装 pre-commit hooks..."
	pre-commit install
	@echo ""
	@echo "✅ Pre-commit hooks 已安装！"
	@echo "现在每次 git commit 前都会自动运行代码检查和格式化"
	@echo ""
	@echo "手动运行所有 hooks: pre-commit run --all-files"

# 运行 pre-commit 检查所有文件
pre-commit-all:
	pre-commit run --all-files

