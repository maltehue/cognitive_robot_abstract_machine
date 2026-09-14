import threading
from typing import List

from coraplex.plans.motion_gate import MotionAndModelChangeGate

# %% bounds

MEETING_TIMEOUT = 5.0
"""
How long a thread waits for the others to reach the same point.
"""

SETTLE_TIME = 0.3
"""
How long to give a thread that should not get through the gate a chance to do so.
"""


def hold_motion(gate: MotionAndModelChangeGate, held: threading.Event) -> None:
    """
    Hold the gate for a motion until told to let go.
    """
    with gate.motion():
        held.wait(timeout=MEETING_TIMEOUT)


# %% motions run next to each other


def test_two_motions_are_held_at_the_same_time():
    gate = MotionAndModelChangeGate()
    both_inside = threading.Barrier(parties=2)
    errors: List[BaseException] = []

    def hold_until_the_other_is_inside() -> None:
        with gate.motion():
            try:
                both_inside.wait(timeout=MEETING_TIMEOUT)
            except threading.BrokenBarrierError as error:
                errors.append(error)

    threads = [
        threading.Thread(target=hold_until_the_other_is_inside, daemon=True)
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=MEETING_TIMEOUT * 2)

    assert errors == []


# %% a model change waits for the running motions


def test_a_model_change_waits_for_a_running_motion():
    gate = MotionAndModelChangeGate()
    let_the_motion_go = threading.Event()
    model_change_entered = threading.Event()
    motion = threading.Thread(
        target=hold_motion, args=(gate, let_the_motion_go), daemon=True
    )
    motion.start()

    def change_the_model() -> None:
        with gate.model_change():
            model_change_entered.set()

    model_change = threading.Thread(target=change_the_model, daemon=True)
    model_change.start()

    assert not model_change_entered.wait(timeout=SETTLE_TIME)

    let_the_motion_go.set()

    assert model_change_entered.wait(timeout=MEETING_TIMEOUT)
    motion.join(timeout=MEETING_TIMEOUT)
    model_change.join(timeout=MEETING_TIMEOUT)


# %% a waiting model change holds back arriving motions


def test_a_motion_arriving_while_a_model_change_waits_is_held_back():
    gate = MotionAndModelChangeGate()
    let_the_first_motion_go = threading.Event()
    let_the_model_change_go = threading.Event()
    model_change_entered = threading.Event()
    second_motion_entered = threading.Event()

    first_motion = threading.Thread(
        target=hold_motion, args=(gate, let_the_first_motion_go), daemon=True
    )
    first_motion.start()

    def change_the_model() -> None:
        with gate.model_change():
            model_change_entered.set()
            let_the_model_change_go.wait(timeout=MEETING_TIMEOUT)

    model_change = threading.Thread(target=change_the_model, daemon=True)
    model_change.start()

    def start_a_second_motion() -> None:
        with gate.motion():
            second_motion_entered.set()

    second_motion = threading.Thread(target=start_a_second_motion, daemon=True)
    second_motion.start()

    assert not second_motion_entered.wait(timeout=SETTLE_TIME)

    let_the_first_motion_go.set()

    assert model_change_entered.wait(timeout=MEETING_TIMEOUT)
    assert not second_motion_entered.wait(timeout=SETTLE_TIME)

    let_the_model_change_go.set()

    assert second_motion_entered.wait(timeout=MEETING_TIMEOUT)
    first_motion.join(timeout=MEETING_TIMEOUT)
    model_change.join(timeout=MEETING_TIMEOUT)
    second_motion.join(timeout=MEETING_TIMEOUT)


# %% the gate is free again afterwards


def test_a_motion_runs_again_after_a_model_change_ended():
    gate = MotionAndModelChangeGate()

    with gate.model_change():
        pass

    entered = threading.Event()
    with gate.motion():
        entered.set()

    assert entered.is_set()
