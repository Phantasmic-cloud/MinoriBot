import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import setproctitle

DRAW_PROCESS_NAME = "MinoriBot-Drawing"


def init_worker_process(name: str | None = None) -> None:
    setproctitle.setproctitle(name or DRAW_PROCESS_NAME)


class ProcessPool:
    _process_pools: list["ProcessPool"] = []

    def __init__(self, max_workers: int, name: str | None = None) -> None:
        self.max_workers = max_workers
        self.name = name or DRAW_PROCESS_NAME
        self.backend = "process"
        self.fallback_error: BaseException | None = None
        try:
            self.executor = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn"),
                initializer=init_worker_process,
                initargs=(self.name,),
            )
        except Exception as e:
            from concurrent.futures import ThreadPoolExecutor

            self.backend = "thread"
            self.fallback_error = e
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        ProcessPool._process_pools.append(self)

    def submit(self, fn, *args, **kwargs):
        return asyncio.get_running_loop().run_in_executor(self.executor, fn, *args, **kwargs)

    @staticmethod
    def shutdown_all() -> None:
        for pool in ProcessPool._process_pools:
            try:
                self_shutdown = getattr(pool.executor, "shutdown")
                self_shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pool.executor.shutdown(wait=False)
        ProcessPool._process_pools.clear()


def is_main_process() -> bool:
    return mp.current_process().name == "MainProcess"


def shutdown_draw_pools() -> None:
    ProcessPool.shutdown_all()