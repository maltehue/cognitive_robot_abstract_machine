import asyncio
from typing import Any, Callable, Coroutine, Optional

# %% running a coroutine from any thread


def run_coroutine(coroutine: Coroutine) -> Any:
    """
    Run a coroutine to completion and return its result.

    The coroutine gets an event loop of its own, so that threads running coroutines at
    the same time do not meet on one loop.

    :param coroutine: The coroutine to run.
    :return: Whatever the coroutine returned.
    """
    event_loop = asyncio.new_event_loop()
    try:
        return event_loop.run_until_complete(coroutine)
    finally:
        event_loop.close()


# %% waiting for a value


async def wait_until_not_none(
    variable_getter: Callable[[], Optional[Any]], check_interval: float = 0.1
) -> Any:
    while variable_getter() is None:
        await asyncio.sleep(check_interval)
    return variable_getter()


async def wait_until_none(
    variable_getter: Callable[[], Optional[Any]], check_interval: float = 0.1
) -> Any:
    while variable_getter() is not None:
        await asyncio.sleep(check_interval)
    return variable_getter()
