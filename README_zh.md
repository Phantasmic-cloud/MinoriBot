# MinoriBot

[English](./README.md) | 中文

<p align="center">
  <img src="./test/mnr.png" alt="Minori" width="280">
</p>

项目形象源于《プロジェクトセカイ カラフルステージ！ feat. 初音ミク》的花里みのり。

基于 OneBot v11 反向 WebSocket 的轻量高性能 Bot 框架，可直接对接 LLOneBot / NapCat / Lagrange / go-cqhttp 等...


## 主要功能

- **自动聊天**：独立 RP‑Agent 驱动角色扮演对话
- **画廊**：图片上传查重，随机抽取查看
- **图片处理** `/img`：抠图、缩放、二维码、语录卡...

详细指令见 [`helps/main.md`](./helps/main.md)。

## 环境

- Python 3.11+
- 一个支持 OneBot v11 反向 WS 的客户端
- 可选：`g++`（编译 imgtool-cpp）、ffmpeg、zbar

## 快速开始

```bash
git clone https://github.com/Phantasmic-cloud/MinoriBot.git
cd MinoriBot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp -a config-example config
```

填写你的基本配置

| 文件 | 填什么 |
| --- | --- |
| `config/global.yaml` | `superuser` 填你的 QQ |

```bash
python bot.py
```
OneBot 客户端反向 WS 指到 `ws://127.0.0.1:8080/onebot/v11/ws`（默认路径，可在 `config/core.yaml` 修改）。

在 bot 所在的群聊发送
```bash
/alive
```

## 目录

```
bot.py              入口
config-example/     配置模板，复制成 config/ 再用
helps/              帮助文档
scripts/            imgtool.cpp 和编译脚本
services/autochat/  自动聊天微服务
src/
  core/             框架：WS、事件、配置、日志
  utils/            指令分发、工具函数
  chat/ llm/        聊天和模型
  gallery/ imgtool/ 画廊和图片
  alive/ status/    存活和状态图
  code/ record/ helper/
```

自动聊天部署见 [`services/autochat/README.md`](./services/autochat/README.md)。

## 声明

本项目的功能实现参考并大量沿用了 [Lunabot](https://github.com/NeuraXmy/lunabot)，意在移除对 NoneBot 的依赖，降低部署难度，注重RP-Agent

## 致谢

功能实现 [Lunabot](https://github.com/NeuraXmy/lunabot)  
状态图 [nonebot-plugin-picstatus](https://github.com/lgc-NB2Dev/nonebot-plugin-picstatus)

## 许可证

[MIT](./LICENSE)