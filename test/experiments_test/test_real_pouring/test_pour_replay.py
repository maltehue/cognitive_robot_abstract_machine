import cv2
import numpy
import pytest

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.stream_detection import ReceiverNotFoundError
from experiments.real_pouring.pour_replay import (
    CameraFrame,
    CupPose,
    NoRecordingFoundError,
    OPENING_HEIGHT,
    PourRecording,
    PourReplay,
    bag_folder_in,
)

from .test_stream_detection import CENTRE, GRAIN_SIZE, _dark_image, _looking_down

DRAIN = ModelPoint(
    repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
)
"""
The drain the replayed bank starts from.
"""

FRAME_PERIOD = 1 / 30
"""
How long apart the recorded frames are, in seconds.
"""

FRAMES_BEFORE_THE_POUR = 15
"""
Frames showing the empty view before the pour starts.
"""

FRAMES_OF_THE_POUR = 30
"""
Frames recorded once the pour has started.
"""

FIRST_GRAIN_FRAME = 9
"""
The first frame of the pour that shows a grain in the opening.
"""


def _encoded(image: numpy.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


def _white_grain() -> numpy.ndarray:
    image = _dark_image()
    image[CENTRE[0] : CENTRE[0] + GRAIN_SIZE, CENTRE[1] : CENTRE[1] + GRAIN_SIZE] = 250
    return image


def _recording() -> PourRecording:
    """
    A camera looking straight down into a receiver whose opening lies at the world
    origin, a source cup tilted beside it, and a grain showing in the opening from
    :data:`FIRST_GRAIN_FRAME` on.
    """
    start = FRAMES_BEFORE_THE_POUR * FRAME_PERIOD
    frames = [
        CameraFrame(stamp=index * FRAME_PERIOD, encoded=_encoded(_dark_image()))
        for index in range(FRAMES_BEFORE_THE_POUR)
    ] + [
        CameraFrame(
            stamp=start + index * FRAME_PERIOD,
            encoded=_encoded(
                _white_grain() if index >= FIRST_GRAIN_FRAME else _dark_image()
            ),
        )
        for index in range(FRAMES_OF_THE_POUR)
    ]
    tilted = cv2.Rodrigues(numpy.array([0.0, 1.2, 0.0]))[0]
    view = _looking_down()
    return PourRecording(
        frames=frames,
        intrinsics=view.intrinsics,
        world_T_camera=view.world_T_camera,
        receiver_position=numpy.array([0.0, 0.0, -OPENING_HEIGHT]),
        cup_poses=[
            CupPose(
                stamp=frame.stamp,
                position=numpy.array([0.1, 0.0, 0.1]),
                rotation=tilted,
            )
            for frame in frames
        ],
        pour_start=start,
    )


def _replay() -> PourReplay:
    """
    The synthetic recording replayed; its view shows no receiving cup, so the opening is
    counted where it projects.
    """
    return PourReplay.of(
        _recording(),
        material=PouredMaterial.STYROFOAM,
        drain=DRAIN,
        settings=BankSettings(tracks_onset=True),
        outline=None,
    )


def test_the_bank_reads_one_frame_in_three_from_the_start_of_the_pour():
    replay = _replay()
    assert [step.frame_index for step in replay.steps] == list(
        range(FRAMES_BEFORE_THE_POUR, FRAMES_BEFORE_THE_POUR + FRAMES_OF_THE_POUR, 3)
    )


def test_the_stream_is_seen_from_the_first_frame_with_a_grain():
    replay = _replay()
    seen = [
        step.frame_index - FRAMES_BEFORE_THE_POUR
        for step in replay.steps
        if step.reading.stream_seen
    ]
    assert seen[0] == FIRST_GRAIN_FRAME


def test_the_grain_mask_shows_the_grain_where_it_lies():
    replay = _replay()
    mask = replay.grain_mask(FRAMES_BEFORE_THE_POUR + FIRST_GRAIN_FRAME)
    assert mask.sum() == GRAIN_SIZE**2
    assert mask[CENTRE[0], CENTRE[1]]


def test_every_step_reports_the_onset_the_hypothesis_in_charge_estimates():
    replay = _replay()
    assert all(step.onset is not None for step in replay.steps)


def test_the_tilt_is_the_cups_angle_from_upright():
    replay = _replay()
    assert replay.steps[0].tilt == pytest.approx(1.2)


# %% a bag inside a zip archive


def test_the_bag_is_the_folder_holding_the_recording(tmp_path):
    bag = tmp_path / "download" / "2026-10-14_17-50-34"
    bag.mkdir(parents=True)
    (bag / "2026-10-14_17-50-34_0.mcap").write_bytes(b"")
    assert bag_folder_in(tmp_path) == bag


def test_a_folder_without_a_recording_says_so(tmp_path):
    with pytest.raises(NoRecordingFoundError):
        bag_folder_in(tmp_path)


def test_a_pour_into_the_translucent_cup_is_replayed_looking_for_it_on_its_mat():
    with pytest.raises(ReceiverNotFoundError):
        PourReplay.of(
            _recording(),
            material=PouredMaterial.CORN,
            drain=DRAIN,
            settings=BankSettings(tracks_onset=True),
            receiver=ReceivingCup.TRANSLUCENT_CUP,
        )
