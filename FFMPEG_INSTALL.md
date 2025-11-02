# FFmpeg 安装指南

FFmpeg 是一个强大的音视频处理工具，对于 YouDub-webui 项目来说是必需的。它用于：
- 修复视频容器格式问题
- 处理音频时间戳
- 视频合成和编码
- 音频格式转换

## 安装方法

### macOS

**方法一：使用 Homebrew（推荐）**

```bash
# 安装 Homebrew（如果还没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 FFmpeg
brew install ffmpeg
```

**方法二：使用 MacPorts**

```bash
sudo port install ffmpeg
```

**验证安装：**

```bash
ffmpeg -version
```

### Linux

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install ffmpeg
```

**CentOS/RHEL/Fedora:**

```bash
# CentOS/RHEL
sudo yum install epel-release
sudo yum install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

**Arch Linux:**

```bash
sudo pacman -S ffmpeg
```

**验证安装：**

```bash
ffmpeg -version
```

### Windows

**方法一：使用 Chocolatey（推荐）**

```powershell
# 以管理员身份运行 PowerShell
# 安装 Chocolatey（如果还没有）
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# 安装 FFmpeg
choco install ffmpeg
```

**方法二：手动安装**

1. 访问 [FFmpeg 官网](https://ffmpeg.org/download.html)
2. 下载 Windows 版本（推荐使用 [BtbN 的构建版本](https://github.com/BtbN/FFmpeg-Builds/releases)）
3. 解压到某个目录（如 `C:\ffmpeg`）
4. 将 `bin` 目录添加到系统 PATH 环境变量：
   - 右键 "此电脑" → "属性" → "高级系统设置" → "环境变量"
   - 在 "系统变量" 中找到 `Path`，点击 "编辑"
   - 添加 FFmpeg 的 `bin` 目录路径（如 `C:\ffmpeg\bin`）
   - 重启终端或重启电脑

**验证安装：**

```cmd
ffmpeg -version
```

### Docker（如果使用 Docker 运行项目）

如果在 Docker 容器中运行，需要在 Dockerfile 中添加：

```dockerfile
# Ubuntu/Debian 基础镜像
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 或者 Alpine Linux 基础镜像
RUN apk add --no-cache ffmpeg
```

## 验证安装

安装完成后，在终端运行以下命令验证：

```bash
ffmpeg -version
```

如果安装成功，会显示 FFmpeg 的版本信息和编译选项。

## 常见问题

### 1. 命令未找到（command not found）

**问题：** 安装了 FFmpeg 但提示找不到命令

**解决方案：**
- 确保 FFmpeg 的 `bin` 目录已添加到 PATH 环境变量
- 重启终端或重新加载 shell 配置：
  ```bash
  # Bash
  source ~/.bashrc
  
  # Zsh
  source ~/.zshrc
  ```

### 2. macOS: Homebrew 安装缓慢

**问题：** Homebrew 下载速度慢

**解决方案：**
使用国内镜像源（如清华大学镜像）：
```bash
# 替换 Homebrew 源
export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git"
export HOMEBREW_CORE_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git"
```

### 3. Linux: 包管理器中的 FFmpeg 版本过旧

**问题：** 系统仓库中的 FFmpeg 版本太旧，缺少某些功能

**解决方案：**

**Ubuntu/Debian - 使用静态构建（推荐）：**

```bash
# 下载最新的静态构建版本
cd /tmp
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
tar xf ffmpeg-git-amd64-static.tar.xz
sudo cp ffmpeg-git-*-static/ffmpeg /usr/local/bin/
sudo cp ffmpeg-git-*-static/ffprobe /usr/local/bin/
sudo chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe
```

**或使用 Snap：**

```bash
sudo snap install ffmpeg
```

### 4. Windows: PATH 环境变量设置后仍无效

**问题：** 设置了 PATH 但命令仍找不到

**解决方案：**
1. 确保添加的是 `bin` 目录的完整路径
2. 关闭并重新打开 PowerShell 或 CMD
3. 如果仍无效，重启电脑

## 在 YouDub-webui 中使用

安装 FFmpeg 后，YouDub-webui 会自动使用它来处理视频和音频。FFmpeg 主要用于：

1. **视频下载后的格式修复** - 修复 MPEG-TS 在 MP4 容器中的问题
2. **音频时间戳修复** - 修复格式错误的 AAC 时间戳
3. **视频合成** - 将视频、音频和字幕合成为最终视频
4. **音频处理** - 音频格式转换和调整

如果遇到视频处理相关的警告或错误，首先确保 FFmpeg 已正确安装并可用。

## 更新 FFmpeg

**macOS (Homebrew):**
```bash
brew upgrade ffmpeg
```

**Linux (APT):**
```bash
sudo apt update && sudo apt upgrade ffmpeg
```

**Windows (Chocolatey):**
```powershell
choco upgrade ffmpeg
```

## 需要帮助？

如果安装过程中遇到问题，可以：

1. 查看 FFmpeg 官方文档：https://ffmpeg.org/documentation.html
2. 提交 GitHub Issue：https://github.com/liuzhao1225/YouDub-webui/issues

