# MinoriBot

English | [中文](./README_zh.md)

<p align="center">
  <img src="./test/mnr.png" alt="Minori" width="280">
</p>

The project mascot is Hanasato Minori from *Project SEKAI COLORFUL STAGE! feat. Hatsune Miku*.

A lightweight, high-performance OneBot v11 reverse-WebSocket bot framework. Drop in LLOneBot / NapCat / Lagrange / go-cqhttp and go.

## Features

- **Autochat**: a separate RP agent for in-group roleplay
- **Gallery**: upload with duplicate checks, pick a random image
- **Image tools** `/img`: cutout, resize, QR codes, quote cards, and more

Command docs live in [`helps/main.md`](./helps/main.md) (Chinese).

## Requirements

- Python 3.11+
- A OneBot v11 reverse-WS client
- Optional: `g++` (imgtool-cpp), ffmpeg, zbar

## Quick start

```bash
git clone https://github.com/Phantasmic-cloud/MinoriBot.git
cd MinoriBot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp -a config-example config
```

Fill in the basics:

| File | What to set |
| --- | --- |
| `config/global.yaml` | `superuser`: your QQ id |

```bash
python bot.py
```

Point the OneBot client reverse WS to `ws://127.0.0.1:8080/onebot/v11/ws` (default; change it in `config/core.yaml`).

In a group the bot is in, send:

```bash
/alive
```

## Layout

```
bot.py              entry
config-example/     config templates; copy to config/
helps/              command help
scripts/            imgtool.cpp and build script
services/autochat/  autochat microservice
src/
  core/             framework: WS, events, config, logging
  utils/            command dispatch, helpers
  chat/ llm/        chat and models
  gallery/ imgtool/ gallery and image tools
  alive/ status/    keepalive and status cards
  code/ record/ helper/
```

Autochat setup: [`services/autochat/README.md`](./services/autochat/README.md).

## Notice

Feature code largely follows [Lunabot](https://github.com/NeuraXmy/lunabot), It removes dependencies on NoneBot to lower deployment barriers, with a primary focus on the RP‑Agent.

## Credits

Features [Lunabot](https://github.com/NeuraXmy/lunabot)  
Status cards [nonebot-plugin-picstatus](https://github.com/lgc-NB2Dev/nonebot-plugin-picstatus)

## License

[MIT](./LICENSE)