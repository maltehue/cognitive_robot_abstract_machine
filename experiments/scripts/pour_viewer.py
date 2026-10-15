"""
Replay a recorded demo pour and look at it frame by frame: what the pole camera saw,
which pixels the stream detector took for grains, and what the live bank believed.

Usage::

    python experiments/scripts/pour_viewer.py <bag folder> [--material corn]
        [--drain fitted] [--onset concept]

The bag is a folder the demo wrote (``RosbagRecording``) or a zip of one; the live bank is run again
on it with the given material, drain and onset handling, and next to it the bank the
robot ran, where the bag carries its trace. Space plays and pauses, the arrow keys step a
frame, the slider jumps.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
import matplotlib
import numpy
from typing_extensions import List, Optional

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.pour_replay import (
    CUP_SIZE,
    FRAME_PERIOD,
    GOAL_FILL,
    GRAINS_PER_FILL,
    PourRecording,
    PourReplay,
    ReplayStep,
)

# Plotting modules of the libraries above switch matplotlib to a backend that draws only
# into files when they are imported; the viewer needs a window, so it chooses one after.
matplotlib.use("QtAgg", force=True)
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# %% choices


class Drain(StrEnum):
    """
    The drains the demo can start the controller from.
    """

    DEMO = "demo"
    """
    The demo's uncalibrated constants.
    """

    FITTED = "fitted"
    """
    The drain fitted to the 2026-10-12 styrofoam pours.
    """

    @property
    def point(self) -> ModelPoint:
        """
        :return: The drain's parameters.
        """
        match self:
            case Drain.DEMO:
                return ModelPoint(
                    repose_angle=0.0,
                    discharge_coefficient=0.2,
                    outflow_rate_constant=0.8,
                )
            case Drain.FITTED:
                return ModelPoint(
                    repose_angle=math.radians(27.0),
                    discharge_coefficient=0.2,
                    outflow_rate_constant=2.03,
                )


class OnsetHandling(StrEnum):
    """
    How the bank treats where the stream starts.
    """

    CONCEPT = "concept"
    """
    Only the onset hypotheses estimate it, as on the robot until 2026-10-14.
    """

    TRACKED = "tracked"
    """
    Every hypothesis estimates it.
    """


CROP_ABOVE, CROP_BELOW, CROP_LEFT, CROP_RIGHT = 230, 170, 300, 170
"""
How far the shown image reaches beyond the opening, in pixels: further up and left, where
the source cup hangs.
"""

GRAIN_COLOUR = numpy.array([255, 0, 255], dtype=numpy.uint8)
"""
The colour detected grain pixels are painted in, BGR.
"""

OPENING_COLOUR = (0, 0, 255)
"""
The colour of the opening's outline, BGR.
"""

PLAYBACK_INTERVAL_MS = int(FRAME_PERIOD * 1000)
"""
How long a frame shows while playing, in milliseconds: real time.
"""


def _lip_angle(fill: float) -> float:
    """
    :return: The tilt at which a cup this full first brings its contents to the lip,
        without a repose angle, in radians.
    """
    return math.atan2(CUP_SIZE.z * (1.0 - fill), CUP_SIZE.x / 2)


# %% the viewer


@dataclass
class PourViewer:
    """
    A window showing one replayed pour.
    """

    replay: PourReplay
    """
    The pour, replayed.
    """

    drain: ModelPoint
    """
    The drain the controller's model started from.
    """

    material: PouredMaterial
    """
    What was poured, whose full cup turns weighed grams into grains.
    """

    title: str
    """
    What the window is called.
    """

    frame_index: int = field(init=False)
    """
    The frame shown.
    """

    playing: bool = field(init=False, default=False)
    """
    Whether the frames advance by themselves.
    """

    _crop: tuple = field(init=False)
    """
    The rows and columns of a frame that are shown: the opening and the source cup above
    it.
    """

    _outline: list = field(init=False)
    """
    The opening's outline in the image, as OpenCV contours.
    """

    _cursors: list = field(init=False, default_factory=list)
    """
    The lines marking the frame shown in every plot.
    """

    def __post_init__(self) -> None:
        recording = self.replay.recording
        self.frame_index = recording.first_pour_frame
        opening = self.replay.opening_mask()
        rows, columns = numpy.nonzero(opening)
        height, width = opening.shape
        self._crop = (
            slice(
                max(0, rows.min() - CROP_ABOVE), min(height, rows.max() + CROP_BELOW)
            ),
            slice(
                max(0, columns.min() - CROP_LEFT),
                min(width, columns.max() + CROP_RIGHT),
            ),
        )
        contours, _ = cv2.findContours(
            opening.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self._outline = contours
        self._build()

    @property
    def _steps(self) -> List[ReplayStep]:
        return self.replay.steps

    def _time_of(self, frame_index: int) -> float:
        return (
            self.replay.recording.frames[frame_index].stamp
            - self.replay.recording.pour_start
        )

    def _step_at(self, frame_index: int) -> Optional[ReplayStep]:
        earlier = [step for step in self._steps if step.frame_index <= frame_index]
        return earlier[-1] if earlier else None

    def _build(self) -> None:
        self.figure = plt.figure(figsize=(16, 9))
        self.figure.canvas.manager.set_window_title(self.title)
        grid = self.figure.add_gridspec(
            4,
            2,
            width_ratios=(1.1, 1),
            left=0.03,
            right=0.98,
            top=0.95,
            bottom=0.12,
            hspace=0.35,
            wspace=0.12,
        )
        self.image_axes = self.figure.add_subplot(grid[:, 0])
        self.image_axes.set_axis_off()
        self.image = self.image_axes.imshow(self._frame_picture(self.frame_index))
        self.caption = self.image_axes.set_title("", fontsize=10, loc="left")
        times = [step.time for step in self._steps]
        degrees = numpy.degrees
        tilt_axes = self.figure.add_subplot(grid[0, 1])
        tilt_axes.plot(
            times, degrees([s.tilt for s in self._steps]), color="black", label="tilt"
        )
        expected = [
            degrees(
                _lip_angle(1.0 - s.reading.counted_grains / GRAINS_PER_FILL)
                + self.drain.repose_angle
                + s.bank.corrections.repose_correction
            )
            for s in self._steps
        ]
        tilt_axes.plot(
            times,
            expected,
            color="tab:orange",
            label="tilt the model expects flow from",
        )
        tilt_axes.set_ylabel("deg")
        tilt_axes.legend(fontsize=8, loc="upper left")
        count_axes = self.figure.add_subplot(grid[1, 1], sharex=tilt_axes)
        count_axes.plot(
            times,
            [s.reading.counted_grains for s in self._steps],
            color="tab:green",
            label="camera count",
        )
        count_axes.plot(
            times,
            [s.bank.received_grains for s in self._steps],
            color="tab:blue",
            label="bank: arrived",
        )
        count_axes.plot(
            times,
            [s.told_grains for s in self._steps],
            color="tab:blue",
            linestyle=":",
            label="told the controller",
        )
        count_axes.axhline(GOAL_FILL * GRAINS_PER_FILL, color="gray", linewidth=0.8)
        weighed = self.replay.recording.weighed
        if weighed is not None:
            full = weighed.get("full_cup_mass_grams", self.material.full_cup_mass_grams)
            count_axes.axhline(
                weighed["received_mass_grams"] / full * GRAINS_PER_FILL,
                color="tab:blue",
                linewidth=0.8,
                linestyle="--",
                label="weighed: arrived",
            )
            count_axes.axhline(
                (weighed["received_mass_grams"] + weighed["spilled_mass_grams"])
                / full
                * GRAINS_PER_FILL,
                color="tab:green",
                linewidth=0.8,
                linestyle="--",
                label="weighed: left the cup",
            )
        live = self.replay.recording.live_trace
        if live is not None:
            start = live[0]["time"]
            count_axes.plot(
                [row["time"] - start for row in live],
                [row["counted_grains"] for row in live],
                color="tab:green",
                alpha=0.35,
                label="camera count on the robot",
            )
        count_axes.set_ylabel("grains")
        count_axes.legend(fontsize=7, loc="upper left")
        probability_axes = self.figure.add_subplot(grid[2, 1], sharex=tilt_axes)
        for hypothesis in self._steps[0].bank.probabilities:
            probability_axes.plot(
                times,
                [s.bank.probabilities[hypothesis] for s in self._steps],
                label=hypothesis,
            )
        probability_axes.set_ylim(-0.02, 1.02)
        probability_axes.set_ylabel("probability")
        probability_axes.legend(fontsize=8, loc="upper left", ncol=3)
        onset_axes = self.figure.add_subplot(grid[3, 1], sharex=tilt_axes)
        controller = [
            degrees(self.drain.repose_angle + s.bank.corrections.repose_correction)
            for s in self._steps
        ]
        onset_axes.plot(
            times,
            controller,
            color="tab:orange",
            label="repose angle the controller runs on",
        )
        mean = numpy.array(
            [numpy.nan if s.onset is None else degrees(s.onset) for s in self._steps]
        )
        spread = numpy.array(
            [
                numpy.nan if s.onset_spread is None else degrees(s.onset_spread)
                for s in self._steps
            ]
        )
        onset_axes.plot(
            times,
            mean,
            color="tab:purple",
            label="estimate of the hypothesis in charge",
        )
        onset_axes.fill_between(
            times, mean - spread, mean + spread, color="tab:purple", alpha=0.2
        )
        if live is not None:
            onset_axes.plot(
                [row["time"] - start for row in live],
                [
                    degrees(
                        self.drain.repose_angle
                        + row["corrections"]["repose_correction"]
                    )
                    for row in live
                ],
                color="tab:orange",
                alpha=0.35,
                label="on the robot",
            )
        onset_axes.set_ylabel("deg")
        onset_axes.set_xlabel("seconds since the pour started")
        onset_axes.legend(fontsize=7, loc="upper left")
        for axes in (tilt_axes, count_axes, probability_axes, onset_axes):
            self._cursors.append(
                axes.axvline(
                    self._time_of(self.frame_index), color="red", linewidth=0.8
                )
            )
        frames = self.replay.recording.frames
        slider_axes = self.figure.add_axes((0.08, 0.04, 0.6, 0.03))
        self.slider = Slider(
            slider_axes,
            "frame",
            self.replay.recording.first_pour_frame,
            len(frames) - 1,
            valinit=self.frame_index,
            valstep=1,
        )
        self.slider.on_changed(lambda value: self._show(int(value)))
        self.play_button = Button(
            self.figure.add_axes((0.72, 0.035, 0.06, 0.04)), "play"
        )
        self.play_button.on_clicked(lambda _: self._toggle())
        back = Button(self.figure.add_axes((0.79, 0.035, 0.04, 0.04)), "<")
        back.on_clicked(lambda _: self._step(-1))
        forward = Button(self.figure.add_axes((0.84, 0.035, 0.04, 0.04)), ">")
        forward.on_clicked(lambda _: self._step(1))
        self._buttons = (back, forward)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.figure.canvas.new_timer(interval=PLAYBACK_INTERVAL_MS)
        self.timer.add_callback(self._advance)
        self._show(self.frame_index)

    def _frame_picture(self, frame_index: int) -> numpy.ndarray:
        image = self.replay.recording.frames[frame_index].image()
        image[self.replay.grain_mask(frame_index)] = GRAIN_COLOUR
        cv2.drawContours(image, self._outline, -1, OPENING_COLOUR, 2)
        return cv2.cvtColor(image[self._crop], cv2.COLOR_BGR2RGB)

    def _show(self, frame_index: int) -> None:
        self.frame_index = frame_index
        self.image.set_data(self._frame_picture(frame_index))
        time = self._time_of(frame_index)
        for cursor in self._cursors:
            cursor.set_xdata([time, time])
        step = self._step_at(frame_index)
        if step is None:
            self.caption.set_text(f"t {time:.2f} s, before the first reading")
        else:
            onset = "-" if step.onset is None else f"{math.degrees(step.onset):.1f}°"
            self.caption.set_text(
                f"t {time:.2f} s   tilt {math.degrees(step.tilt):.1f}°   stream {'seen' if step.reading.stream_seen else 'not seen'}   "
                f"count {step.reading.counted_grains:.1f}\n"
                f"bank: {step.bank.active} in charge, arrived {step.bank.received_grains:.1f}, told {step.told_grains:.1f}, "
                f"onset estimate {onset}, repose correction {math.degrees(step.bank.corrections.repose_correction):+.1f}°"
            )
        if int(self.slider.val) != frame_index:
            self.slider.set_val(frame_index)
        self.figure.canvas.draw_idle()

    def _step(self, frames: int) -> None:
        last = len(self.replay.recording.frames) - 1
        self._show(
            min(
                last,
                max(self.replay.recording.first_pour_frame, self.frame_index + frames),
            )
        )

    def _advance(self) -> None:
        if self.frame_index >= len(self.replay.recording.frames) - 1:
            self._toggle()
            return
        self._step(1)

    def _toggle(self) -> None:
        self.playing = not self.playing
        self.play_button.label.set_text("pause" if self.playing else "play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_key(self, event) -> None:
        match event.key:
            case " ":
                self._toggle()
            case "right":
                self._step(1)
            case "left":
                self._step(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "bag",
        type=Path,
        help="a bag folder the demo wrote, or a zip archive holding one",
    )
    parser.add_argument(
        "--material",
        type=PouredMaterial,
        default=PouredMaterial.STYROFOAM,
        choices=list(PouredMaterial),
    )
    parser.add_argument(
        "--receiver",
        type=ReceivingCup,
        default=ReceivingCup.JEROEN_CUP,
        choices=list(ReceivingCup),
    )
    parser.add_argument("--drain", type=Drain, default=Drain.DEMO, choices=list(Drain))
    parser.add_argument(
        "--onset",
        type=OnsetHandling,
        default=OnsetHandling.TRACKED,
        choices=list(OnsetHandling),
    )
    arguments = parser.parse_args()
    print(f"reading {arguments.bag} ...", flush=True)
    recording = PourRecording.read(arguments.bag)
    print(
        f"replaying {len(recording.frames)} frames through the detector and the bank ...",
        flush=True,
    )
    replay = PourReplay.of(
        recording,
        material=arguments.material,
        drain=arguments.drain.point,
        settings=BankSettings(tracks_onset=arguments.onset is OnsetHandling.TRACKED),
        receiver=arguments.receiver,
    )
    viewer = PourViewer(
        replay=replay,
        drain=arguments.drain.point,
        material=arguments.material,
        title=f"{arguments.bag.name}: {arguments.material}, {arguments.drain} drain, onset {arguments.onset}",
    )
    plt.show()
    # Matplotlib holds its widgets' and timer's callbacks weakly; the viewer has to live
    # until the window closes for its buttons, keys and playback to keep working.
    del viewer


if __name__ == "__main__":
    main()
