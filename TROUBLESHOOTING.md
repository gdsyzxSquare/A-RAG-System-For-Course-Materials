# 常见问题解决方案 (Troubleshooting)

## ❌ 错误: `Client.__init__() got an unexpected keyword argument 'proxies'`

### 问题描述
运行 `test_deepseek.py` 或使用RAG系统时出现以下错误：
```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

### 原因
这是由于 `httpx` 包版本不兼容导致的。OpenAI SDK 1.6.1 需要 `httpx==0.25.x` 版本，但如果安装了更新的 `httpx 0.28.x`，会导致API冲突。

### 解决方案

**方法1: 降级 httpx (推荐)**
```bash
pip uninstall httpx -y
pip install httpx==0.25.2
```

**方法2: 升级整个OpenAI SDK**
```bash
pip install --upgrade openai httpx
```

注意：方法2可能需要更新代码以适配新版API。

### 验证修复
运行测试脚本验证问题已解决：
```bash
python test_simple.py
python test_deepseek.py
```

## ❌ 错误: `Invalid API key`

### 问题描述
```
Error: Invalid API key
```

### 解决方案
1. 检查 `.env` 文件中的API key是否正确
2. 确保API key有效（在DeepSeek平台验证）
3. 检查是否有多余的空格或引号
4. 确保文件名是 `.env` 而不是 `.env.example`

### 配置示例
```bash
# .env 文件内容
DEEPSEEK_API_KEY=sk-your-actual-key-here
```

## ❌ 错误: `Connection timeout` 或 `Connection refused`

### 问题描述
```
Error: Connection timeout
Error: Connection refused
```

### 可能原因
1. 网络连接问题
2. 防火墙/代理设置
3. DeepSeek API服务暂时不可用

### 解决方案

**1. 检查网络连接**
```bash
ping api.deepseek.com
```

**2. 清除代理设置**
如果有代理设置干扰，临时清除：
```bash
# Windows PowerShell
$env:HTTP_PROXY=""
$env:HTTPS_PROXY=""

# 然后重新运行测试
python test_deepseek.py
```

**3. 检查防火墙**
- 确保防火墙允许访问 `https://api.deepseek.com`
- 临时关闭防火墙测试

## ❌ 错误: `Rate limit exceeded`

### 问题描述
```
Error: Rate limit exceeded
```

### 解决方案
1. 降低请求频率
2. 等待几分钟后重试
3. 升级API套餐（如适用）
4. 在代码中添加重试延迟

## ❌ 错误: 无法找到 `.env` 文件

### 问题描述
程序无法加载环境变量，API key为None

### 解决方案
1. 确保 `.env` 文件在项目根目录
2. 从模板创建：
```bash
copy .env.example .env
```
3. 编辑 `.env` 添加你的API key

## ❌ 错误: `ModuleNotFoundError: No module named 'openai'`

### 问题描述
```
ModuleNotFoundError: No module named 'openai'
```

### 解决方案
安装所需依赖：
```bash
pip install -r requirements.txt
```

或单独安装：
```bash
pip install openai==1.6.1 httpx==0.25.2
```

## 📝 完整依赖列表

本项目关键依赖及其版本：

```
openai==1.6.1
httpx==0.25.2
python-dotenv==1.0.0
sentence-transformers==2.3.1
faiss-cpu==1.7.4
rank-bm25==0.2.2
```

## 🔍 调试技巧

### 1. 测试API连接
```bash
python test_simple.py
```

### 2. 检查包版本
```bash
pip show openai httpx
```

### 3. 查看详细错误
在Python代码中添加：
```python
import traceback
try:
    # your code
except Exception as e:
    traceback.print_exc()
```

### 4. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📧 获取帮助

如果以上方法都无法解决问题：

1. 查看 `DEEPSEEK_SETUP.md` 获取详细设置说明
2. 检查 DeepSeek 官方文档: https://platform.deepseek.com/api-docs/
3. 验证Python版本 (建议 Python 3.9+):
   ```bash
   python --version
   ```

## ✅ 快速自检清单

遇到问题时，按顺序检查：

- [ ] Python版本 >= 3.9
- [ ] 已安装所有依赖 (`pip install -r requirements.txt`)
- [ ] httpx版本 == 0.25.2
- [ ] `.env` 文件存在且包含正确的API key
- [ ] 网络可以访问 `api.deepseek.com`
- [ ] 没有代理设置干扰
- [ ] API key在DeepSeek平台是有效的
- [ ] 运行 `python test_simple.py` 测试通过

如果所有项都通过，系统应该可以正常工作！
