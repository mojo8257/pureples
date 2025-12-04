# tests/conftest.py
import multiprocessing as mp
import os
import sys
import time
import pytest


@pytest.fixture(autouse=True, scope="session")
def _mp_leak_guard_session():
    """
    会话级保险丝：任一测试失败或异常路径未显式回收进程池时，
    在会话结束统一终止所有遗留的子进程，避免 pytest 卡住。
    """
    yield
    leaked = mp.active_children()
    if leaked:
        sys.stderr.write(
            f"\n[leak-guard][session] found {len(leaked)} active children; terminating...\n"
        )
        for p in leaked:
            try:
                p.terminate()
            except Exception:
                pass
        # 等待一小会儿
        deadline = time.time() + 5.0
        for p in leaked:
            try:
                timeout = max(0.0, deadline - time.time())
                p.join(timeout=timeout)
            except Exception:
                pass


@pytest.fixture(autouse=True, scope="module")
def _mp_leak_guard_module():
    """
    模块级保险丝：每个测试模块结束后再清一次，进一步降低串行模块间的干扰。
    """
    yield
    leaked = mp.active_children()
    if leaked:
        sys.stderr.write(
            f"\n[leak-guard][module] found {len(leaked)} active children; terminating...\n"
        )
        for p in leaked:
            try:
                p.terminate()
            except Exception:
                pass
        deadline = time.time() + 5.0
        for p in leaked:
            try:
                timeout = max(0.0, deadline - time.time())
                p.join(timeout=timeout)
            except Exception:
                pass
