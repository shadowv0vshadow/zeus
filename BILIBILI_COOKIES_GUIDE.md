# Bilibili Cookie 获取指南

为了上传视频到 Bilibili，您需要获取 `BILI_SESSDATA` 和 `BILI_BILI_JCT` 这两个 Cookie 值。

## 快速开始

1. **登录 Bilibili**：在浏览器中登录您的 Bilibili 账号
2. **打开开发者工具**：按 `F12` 或右键点击页面 → "检查" / "Inspect"
3. **查看 Cookies**：在开发者工具中找到 Cookies
4. **复制 Cookie 值**：复制 `SESSDATA` 和 `bili_jct` 的值
5. **设置环境变量**：将值添加到 `.env` 文件中

## 详细步骤

### 方法一：使用浏览器开发者工具（推荐）

#### Chrome/Edge 浏览器

1. **打开 Bilibili 并登录**
   - 访问 https://www.bilibili.com
   - 确保已登录您的账号

2. **打开开发者工具**
   - 按 `F12` 键，或
   - 右键点击页面 → 选择"检查"（Inspect）

3. **切换到 Application 标签**
   - 在开发者工具顶部，点击 "Application"（应用程序）标签
   - 如果没有看到，可能需要点击 ">>" 查看更多标签

4. **查看 Cookies**
   - 在左侧面板中，展开 "Storage"（存储）→ "Cookies"
   - 点击 `https://www.bilibili.com`
   - 在右侧会显示所有 Cookie

5. **找到并复制 Cookie 值**
   - 找到名为 `SESSDATA` 的 Cookie，复制其 **Value**（值）
   - 找到名为 `bili_jct` 的 Cookie，复制其 **Value**（值）

6. **设置环境变量**
   - 在项目根目录找到 `.env` 文件（如果没有，从 `.env.example` 复制）
   - 添加以下两行：
   ```bash
   BILI_SESSDATA=你复制的SESSDATA值
   BILI_BILI_JCT=你复制的bili_jct值
   ```
   **注意**：值不需要引号，直接粘贴即可

#### Firefox 浏览器

1. **打开 Bilibili 并登录**
   - 访问 https://www.bilibili.com
   - 确保已登录您的账号

2. **打开开发者工具**
   - 按 `F12` 键，或
   - 右键点击页面 → 选择"检查元素"

3. **切换到存储标签**
   - 在开发者工具顶部，点击 "存储"（Storage）标签

4. **查看 Cookies**
   - 在左侧面板中，展开 "Cookie"
   - 点击 `https://www.bilibili.com`
   - 在右侧会显示所有 Cookie

5. **找到并复制 Cookie 值**
   - 找到 `SESSDATA` 和 `bili_jct` 的值并复制

6. **设置环境变量**
   - 在 `.env` 文件中添加：
   ```bash
   BILI_SESSDATA=你复制的SESSDATA值
   BILI_BILI_JCT=你复制的bili_jct值
   ```

#### Safari 浏览器

1. **启用开发者工具**
   - Safari → 偏好设置 → 高级 → 勾选"在菜单栏中显示开发菜单"

2. **打开 Bilibili 并登录**
   - 访问 https://www.bilibili.com
   - 确保已登录您的账号

3. **打开 Web 检查器**
   - 开发 → 显示 Web 检查器
   - 按 `Cmd + Option + I`

4. **切换到存储标签**
   - 点击 "存储"（Storage）标签
   - 在左侧展开 "Cookie" → `https://www.bilibili.com`

5. **找到并复制 Cookie 值**

6. **设置环境变量**

### 方法二：使用浏览器扩展（适用于 Chrome/Edge）

如果手动查找 Cookie 比较困难，可以使用浏览器扩展：

1. **安装扩展**
   - Chrome/Edge: 搜索并安装 "Cookie-Editor" 或 "EditThisCookie"
   - 这些扩展可以更方便地查看和导出 Cookies

2. **使用扩展**
   - 点击扩展图标
   - 找到 `SESSDATA` 和 `bili_jct`
   - 复制它们的值

### 方法三：使用命令行工具（高级）

如果您熟悉命令行，可以使用以下工具：

#### Chrome (macOS)
```bash
# 查看 Chrome Cookies（需要先退出 Chrome）
sqlite3 ~/Library/Application\ Support/Google/Chrome/Default/Cookies \
  "SELECT name, value FROM cookies WHERE host_key LIKE '%bilibili.com%' AND name IN ('SESSDATA', 'bili_jct');"
```

#### Chrome (Linux)
```bash
sqlite3 ~/.config/google-chrome/Default/Cookies \
  "SELECT name, value FROM cookies WHERE host_key LIKE '%bilibili.com%' AND name IN ('SESSDATA', 'bili_jct');"
```

## 环境变量设置

### 设置方式一：使用 .env 文件（推荐）

1. **创建或编辑 .env 文件**
   ```bash
   # 如果不存在，从模板复制
   cp .env.example .env
   
   # 编辑 .env 文件
   nano .env  # 或使用其他编辑器
   ```

2. **添加 Cookie 值**
   ```bash
   # Bilibili 上传所需的 Cookie
   BILI_SESSDATA=你的SESSDATA值（很长的一串字符）
   BILI_BILI_JCT=你的bili_jct值（通常是32位字符）
   ```

3. **保存文件**
   - 确保没有引号包裹值
   - 确保每行一个变量

### 设置方式二：使用环境变量（临时）

#### macOS/Linux
```bash
export BILI_SESSDATA="你的SESSDATA值"
export BILI_BILI_JCT="你的bili_jct值"
```

#### Windows (PowerShell)
```powershell
$env:BILI_SESSDATA="你的SESSDATA值"
$env:BILI_BILI_JCT="你的bili_jct值"
```

#### Windows (CMD)
```cmd
set BILI_SESSDATA=你的SESSDATA值
set BILI_BILI_JCT=你的bili_jct值
```

## Cookie 值示例格式

**SESSDATA** 示例：
```
abc123def456ghi789jkl012mno345pqr678stu901vwx234yz567==
```
- 通常是一串 Base64 编码的长字符串
- 可能包含等号（=）结尾

**bili_jct** 示例：
```
a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```
- 通常是32位十六进制字符
- 不包含特殊符号

## 验证设置

设置完成后，可以运行以下命令验证：

```bash
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('SESSDATA:', os.getenv('BILI_SESSDATA')[:20] + '...' if os.getenv('BILI_SESSDATA') else '未设置'); print('bili_jct:', os.getenv('BILI_BILI_JCT')[:10] + '...' if os.getenv('BILI_BILI_JCT') else '未设置')"
```

或者在代码中测试：
```python
from dotenv import load_dotenv
import os

load_dotenv()

sessdata = os.getenv('BILI_SESSDATA')
bili_jct = os.getenv('BILI_BILI_JCT')

if sessdata and bili_jct:
    print("✅ Cookies 已设置")
    print(f"SESSDATA: {sessdata[:20]}...")
    print(f"bili_jct: {bili_jct[:10]}...")
else:
    print("❌ Cookies 未设置")
```

## 安全注意事项

⚠️ **重要提示**：

1. **不要泄露 Cookie**
   - Cookie 相当于您的登录凭证
   - 不要将 `.env` 文件提交到 Git
   - 不要分享您的 Cookie 值

2. **定期更新 Cookie**
   - Cookie 会过期，通常几周到几个月
   - 如果上传失败，可能是 Cookie 过期，需要重新获取

3. **使用个人账号**
   - 建议使用个人测试账号
   - 不要在重要账号上使用自动化上传

4. **Git 忽略**
   - 确保 `.env` 文件在 `.gitignore` 中
   - 不要将包含真实 Cookie 的 `.env` 文件推送到仓库

## 常见问题

### Q1: Cookie 值有多长？

**A**: 
- `SESSDATA` 通常很长（几百个字符），可能包含等号（=）
- `bili_jct` 通常是32位字符（十六进制）

### Q2: Cookie 多久过期？

**A**: 
- 通常几周到几个月不等
- 如果长期未使用 Bilibili，Cookie 可能会失效
- 建议定期检查并更新

### Q3: 上传时提示 Cookie 错误？

**A**: 
- 检查 Cookie 是否正确复制（没有多余空格）
- 检查 Cookie 是否过期（重新登录并获取新的）
- 确保 `.env` 文件在项目根目录
- 确保应用已重启（加载新的环境变量）

### Q4: 可以同时使用多个账号吗？

**A**: 
- 当前设计只支持一个账号
- 如果需要切换账号，修改 `.env` 文件中的值并重启应用

### Q5: Cookie 从哪里获取？

**A**: 
- 必须从已登录 Bilibili 的浏览器中获取
- 确保登录的是您想要上传视频的账号
- 建议使用个人测试账号

## 获取流程总结

```
1. 登录 Bilibili (https://www.bilibili.com)
   ↓
2. 打开开发者工具 (F12)
   ↓
3. 切换到 Application/存储 标签
   ↓
4. 找到 Cookies → https://www.bilibili.com
   ↓
5. 复制 SESSDATA 和 bili_jct 的值
   ↓
6. 添加到 .env 文件
   ↓
7. 重启应用
```

## 需要帮助？

如果遇到问题，可以：
1. 检查 [bilibili-toolman 文档](https://github.com/mos9527/bilibili-toolman)
2. 查看项目 Issues：https://github.com/liuzhao1225/YouDub-webui/issues
3. 确保已安装 `python-dotenv`：`pip install python-dotenv`

