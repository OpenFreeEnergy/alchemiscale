from time import sleep
import requests
import contextlib
import multiprocessing


@contextlib.contextmanager
def running_service(target, port, args):
    multiprocessing.set_start_method("fork", force=True)
    proc = multiprocessing.Process(target=target, args=args, daemon=True)
    proc.start()

    if not proc.is_alive():
        raise RuntimeError("The test server could not be started.")

    timeout = True
    for _ in range(40):
        try:
            ping = requests.get(f"http://127.0.0.1:{port}/ping")
            ping.raise_for_status()
        except IOError:
            sleep(0.25)
            continue
        timeout = False
        break
    if timeout:
        raise RuntimeError("The test server could not be reached.")

    yield

    proc.terminate()
    while proc.is_alive():
        sleep(0.1)
