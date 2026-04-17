import time
from typing import Any, Callable

def timer(func: Callable) -> Callable:

    def wrap(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end-start
    return wrap
