# 安装 Chinese-CLIP 的方法（解决 Windows 编译问题）

## 方法 1: 安装预编译的 lmdb wheel

```bash
# 从这个网站下载适合你 Python 版本的 lmdb wheel
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#lmdb
# 例如 Python 3.10: lmdb-1.3.0-cp310-cp310-win_amd64.whl

# 下载后安装
pip install 下载的wheel文件路径/lmdb-1.3.0-cp310-cp310-win_amd64.whl

# 然后安装 cn-clip
pip install cn-clip
```

## 方法 2: 跳过 lmdb，直接安装核心包

```bash
# 安装 cn-clip 的核心依赖，跳过 lmdb
pip install timm transformers

# 修改 clip_ranker.py，手动处理导入
```

## 方法 3: 使用 WSL 或 Linux（最佳）

如果有条件，在 Linux 环境中 Chinese-CLIP 安装非常简单：
```bash
pip install cn-clip
```
