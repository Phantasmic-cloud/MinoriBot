import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 相对路径落到项目根

from src.core import CoreConfig, get_logger, load_plugins, serve, setup_logging
from src.draw.process_pool import shutdown_draw_pools
from src.utils.scheduler import start_scheduler, stop_scheduler
from src.utils.utils import close_client_session, shutdown_thread_pool

logger = get_logger("main")
PROCESS_NAME = "MinoriBot-main"

_BANNER = r"""
███╗   ███╗ ██╗ ███╗   ██╗  ██████╗  ██████╗  ██╗ ██████╗   ██████╗  ████████╗
████╗ ████║ ██║ ████╗  ██║ ██╔═══██╗ ██╔══██╗ ██║ ██╔══██╗ ██╔═══██╗ ╚══██╔══╝
██╔████╔██║ ██║ ██╔██╗ ██║ ██║   ██║ ██████╔╝ ██║ ██████╔╝ ██║   ██║    ██║
██║╚██╔╝██║ ██║ ██║╚██╗██║ ██║   ██║ ██╔══██╗ ██║ ██╔══██╗ ██║   ██║    ██║
██║ ╚═╝ ██║ ██║ ██║ ╚████║ ╚██████╔╝ ██║  ██║ ██║ ██████╔╝ ╚██████╔╝    ██║
╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═══╝  ╚═════╝  ╚═╝  ╚═╝ ╚═╝ ╚═════╝   ╚═════╝     ╚═╝
""".strip("\n")
_BANNER_COLOR = "\033[38;2;255;204;170m"
_RESET = "\033[0m"


def _print_banner(color: bool = True) -> None:
    """启动时打印 MINORIBOT banner。"""
    use_color = color
    if os.environ.get("NO_COLOR"):
        use_color = False
    elif os.environ.get("FORCE_COLOR"):
        use_color = True
    elif not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        use_color = False
    text = f"{_BANNER_COLOR}{_BANNER}{_RESET}" if use_color else _BANNER
    print(f"\n{text}\n", flush=True)


def _set_process_name(name: str) -> None:
    """把进程名改成 MinoriBot-main，方便 ps 辨认。"""
    try:
        import setproctitle

        setproctitle.setproctitle(name)
        return
    except Exception:
        pass
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        libc.prctl(15, name.encode(), 0, 0, 0)
    except Exception:
        pass


async def run(config_dir: str | None = None) -> None:
    """加载配置和插件，启动反向 WS，退出时关掉调度器、绘图进程和 HTTP session。"""
    config = CoreConfig.load(config_dir)
    setup_logging(config.log_level, color=config.log_color)
    _print_banner(color=config.log_color)
    logger.info("=========== MinoriBot Client v0.1.0 ==========")
    _set_process_name(PROCESS_NAME)
    load_plugins()
    start_scheduler()
    try:
        await serve(config)
    finally:
        stop_scheduler()
        shutdown_draw_pools()
        await close_client_session()
        shutdown_thread_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description="MinoriBot / OneBot v11 反向 WS 服务端")
    parser.add_argument("-c", "--config-dir", default=str(ROOT / "config"), help="配置目录，默认 ./config")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.config_dir))
    except KeyboardInterrupt:
        logger.info("Stopped...")


if __name__ == "__main__":
    main()