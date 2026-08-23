import threading
import time

from core.tui import BenchmarkTui


def test_q_requests_cancellation_while_running():
    tui = BenchmarkTui(enabled=False)

    tui._HandleKey("q")

    assert tui.cancelRequested
    assert tui.quitRequested


def test_q_only_quits_after_benchmark_finished():
    tui = BenchmarkTui(enabled=False)
    tui.finished = True

    tui._HandleKey("q")

    assert not tui.cancelRequested
    assert tui.quitRequested


def test_noninteractive_final_dashboard_does_not_wait():
    tui = BenchmarkTui(enabled=False)

    tui.FinishAndWait()

    assert not tui.finished


def test_interactive_final_dashboard_waits_for_q():
    tui = BenchmarkTui(enabled=False)
    tui.enabled = True
    tui._inputEnabled = True
    waiter = threading.Thread(target=tui.FinishAndWait)

    waiter.start()
    for _ in range(100):
        if tui.finished:
            break
        time.sleep(0.001)

    assert tui.finished
    assert waiter.is_alive()
    tui._HandleKey("q")
    waiter.join(timeout=0.2)
    assert not waiter.is_alive()
    assert not tui.cancelRequested
