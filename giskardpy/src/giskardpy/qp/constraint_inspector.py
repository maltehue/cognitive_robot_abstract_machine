"""
Scrubbing viewer for a recorded motion, showing what every constraint asked of every
degree of freedom in the control cycle the slider points at.

Run it on a recording written by
:class:`~giskardpy.qp.control_cycle_recording.ControlCycleRecorder`::

    python -m giskardpy.qp.constraint_inspector /tmp/control_cycles/goal_0.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Another workspace package binds matplotlib's Qt backend to PySide6 when it is imported
# first, while this viewer is built on the PyQt5 that giskardpy depends on.
os.environ.setdefault("QT_API", "pyqt5")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QShortcut,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from giskardpy.qp.control_cycle_analysis import ControlCycleAnalysis
from giskardpy.qp.control_cycle_recording import (
    NODE_INDEX_SEPARATOR,
    ControlCycleRecording,
)

if TYPE_CHECKING:
    from giskardpy.qp.recorded_state_publisher import RecordedWorldStatePublisher

PLAYBACK_INTERVAL_IN_MILLISECONDS = 50
"""
Delay between two cycles while the recording plays.
"""

LABEL_FONT_SIZE = 7
"""
Font size of the row and column labels of the charts.
"""

TITLE_FONT_SIZE = 9
"""
Font size of the chart titles.
"""

SUBTITLE_FONT_SIZE = 7
"""
Font size of the line that explains how to read a chart.
"""

MAXIMUM_LABEL_LENGTH = 26
"""
Longest label drawn next to an axis before its middle is elided.

The untruncated names stay readable in the table of the inspected cycle.
"""

VISIBLE_COST_DECADES = 6
"""
How many powers of ten below the largest cost the timeline still tells apart.

Constraint weights span orders of magnitude, so on a linear scale the most expensive
constraint hides every other one.
"""

ELLIPSIS = "…"
"""
Marks where the middle of an over-long label was cut out.
"""

# %% palette


@dataclass(frozen=True)
class ChartPalette:
    """
    The colors of the viewer, taken unchanged from the documented reference palette.

    Every color does exactly one job: identity for the task a constraint belongs to,
    magnitude for costs, polarity for signed quantities, and ink for everything that is
    not data.
    """

    surface: str = "#fcfcfb"
    """
    Background of every chart.
    """

    primary_ink: str = "#0b0b0b"
    """
    Color of titles and other text that must be read first.
    """

    secondary_ink: str = "#52514e"
    """
    Color of supporting text.
    """

    muted_ink: str = "#898781"
    """
    Color of axis labels and ticks, which stay recessive.
    """

    gridline: str = "#e1e0d9"
    """
    Color of the hairline grid.
    """

    limit: str = "#d03b3b"
    """
    Color that marks a limit being approached, paired with a label wherever it is used.
    """

    task_colors: tuple[str, ...] = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    )
    """
    Identity colors of the tasks, assigned in this fixed order and never cycled.
    """

    overflow_task_color: str = "#898781"
    """
    Color of the tasks beyond the eighth, which share one bucket.
    """

    sequential_steps: tuple[str, ...] = (
        "#fcfcfb",
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    )
    """
    One-hue ramp from near zero to the largest magnitude.

    The lightest end is the chart surface, so a constraint that cost nothing leaves no
    mark at all and the eye only lands on the ones that did.
    """

    diverging_steps: tuple[str, ...] = ("#e34948", "#f0efec", "#2a78d6")
    """
    Two opposite hues around a neutral midpoint, for quantities that carry a sign.
    """

    @property
    def magnitude_color_map(self) -> LinearSegmentedColormap:
        """
        Color map for quantities that only grow.
        """
        return LinearSegmentedColormap.from_list(
            "magnitude", list(self.sequential_steps)
        )

    @property
    def polarity_color_map(self) -> LinearSegmentedColormap:
        """
        Color map for quantities whose sign is meaningful.
        """
        return LinearSegmentedColormap.from_list("polarity", list(self.diverging_steps))


# %% labels


def _elide_middle(text: str, maximum_length: int = MAXIMUM_LABEL_LENGTH) -> str:
    """
    Shortens a label from the middle, so both of its ends stay readable.
    """
    if len(text) <= maximum_length:
        return text
    head_length = (maximum_length - 1) // 2
    tail_length = maximum_length - 1 - head_length
    return f"{text[:head_length]}{ELLIPSIS}{text[len(text) - tail_length:]}"


@dataclass
class ChartLabels:
    """
    Short names and identity colors for the rows and degrees of freedom of a recording.

    The names a motion statechart generates are built for uniqueness rather than for
    reading, so they are shortened here and kept in full in the table of the cycle.
    """

    recording: ControlCycleRecording
    """
    The recording whose rows and degrees of freedom are labelled.
    """

    palette: ChartPalette = field(default_factory=ChartPalette)
    """
    Supplies the identity colors of the tasks.
    """

    @property
    def task_names(self) -> list[str]:
        """
        The task every row belongs to, without the index that makes nodes unique when
        the name alone already is.
        """
        node_names = self.recording.structure.node_names
        without_index = [name.split(NODE_INDEX_SEPARATOR)[0] for name in node_names]
        is_ambiguous = len(set(without_index)) < len(set(node_names))
        return node_names if is_ambiguous else without_index

    @property
    def ordered_task_names(self) -> list[str]:
        """
        The distinct tasks in the order their first row appears.
        """
        ordered: list[str] = []
        for task_name in self.task_names:
            if task_name not in ordered:
                ordered.append(task_name)
        return ordered

    def task_color(self, task_name: str) -> str:
        """
        The identity color of a task, shared by everything past the eighth.
        """
        position = self.ordered_task_names.index(task_name)
        if position >= len(self.palette.task_colors):
            return self.palette.overflow_task_color
        return self.palette.task_colors[position]

    @property
    def row_colors(self) -> list[str]:
        """
        The identity color of every row's task.
        """
        return [self.task_color(task_name) for task_name in self.task_names]

    @property
    def row_labels(self) -> list[str]:
        """
        Short label of every row: its task, numbered when the task owns several rows.
        """
        task_names = self.task_names
        counts = {name: task_names.count(name) for name in set(task_names)}
        seen: dict[str, int] = {}
        labels = []
        for task_name in task_names:
            index = seen.get(task_name, 0)
            seen[task_name] = index + 1
            labels.append(
                f"{task_name} {index}" if counts[task_name] > 1 else task_name
            )
        return [_elide_middle(label) for label in labels]

    @property
    def degree_of_freedom_labels(self) -> list[str]:
        """
        Short label of every degree of freedom, without its robot prefix and joint
        suffix.
        """
        labels = []
        for name in self.recording.structure.degree_of_freedom_names:
            segments = name.split("/")
            tail = "/".join(segments[1:]) if len(segments) > 1 else name
            labels.append(_elide_middle(tail.removesuffix("_joint")))
        return labels


# %% panels


class InspectorPanel(ABC):
    """
    One chart of the viewer, redrawn whenever the inspected control cycle changes.
    """

    def __init__(self, analysis: ControlCycleAnalysis, labels: ChartLabels) -> None:
        self.analysis = analysis
        self.labels = labels
        self.palette = labels.palette
        self.figure = Figure(layout="constrained", facecolor=self.palette.surface)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot()
        self.axes.set_facecolor(self.palette.surface)

    def _create_axes_with_task_strip(self) -> None:
        """
        Replace the plain axes by a chart with a strip of task colors down its side.

        The strip carries the identity of each row, which leaves the row labels in ink
        where they stay readable.
        """
        self.figure.clear()
        strip_axes, self.axes = self.figure.subplots(
            1, 2, width_ratios=[1, 28], gridspec_kw={"wspace": 0.03}
        )
        self.axes.set_facecolor(self.palette.surface)
        row_colors = self.labels.row_colors
        strip_axes.imshow(
            np.arange(len(row_colors)).reshape(-1, 1),
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap(row_colors),
        )
        strip_axes.set_xticks([])
        row_labels = self.labels.row_labels
        strip_axes.set_yticks(range(len(row_labels)))
        strip_axes.set_yticklabels(
            row_labels, fontsize=LABEL_FONT_SIZE, color=self.palette.secondary_ink
        )
        strip_axes.tick_params(length=0)
        for spine in strip_axes.spines.values():
            spine.set_visible(False)

    @property
    def recording(self) -> ControlCycleRecording:
        """
        The recording being inspected.
        """
        return self.analysis.recording

    @abstractmethod
    def draw(self, cycle_index: int) -> None:
        """
        Draw the panel for the given control cycle.
        """

    def _set_heading(self, title: str, subtitle: str) -> None:
        """
        Name what the chart shows and say in one line how to read it.
        """
        self.figure.suptitle(
            title,
            fontsize=TITLE_FONT_SIZE,
            color=self.palette.primary_ink,
            ha="left",
            x=0.01,
        )
        self.axes.set_title(
            subtitle, fontsize=SUBTITLE_FONT_SIZE, color=self.palette.muted_ink, pad=4
        )

    def _style_axes(self) -> None:
        """
        Push the frame and the ticks behind the data.
        """
        self.axes.tick_params(
            colors=self.palette.muted_ink, labelsize=LABEL_FONT_SIZE, length=0
        )
        for spine in self.axes.spines.values():
            spine.set_visible(False)

    def _add_scale(self, image, label: str) -> None:
        """
        Add the scale that says what the colors of a chart mean.
        """
        color_bar = self.figure.colorbar(image, ax=self.axes, pad=0.02, fraction=0.05)
        color_bar.set_label(
            label, fontsize=SUBTITLE_FONT_SIZE, color=self.palette.muted_ink
        )
        color_bar.ax.tick_params(
            colors=self.palette.muted_ink, labelsize=LABEL_FONT_SIZE, length=0
        )
        color_bar.outline.set_visible(False)


class ViolationTimelinePanel(InspectorPanel):
    """
    Shows what every constraint cost the optimizer over the whole motion.
    """

    def __init__(self, analysis: ControlCycleAnalysis, labels: ChartLabels) -> None:
        super().__init__(analysis, labels)
        self._create_axes_with_task_strip()
        costs = self.analysis.slack_costs.T
        image = self.axes.imshow(
            costs,
            aspect="auto",
            interpolation="nearest",
            cmap=self.palette.magnitude_color_map,
            norm=self._cost_scale(costs),
            extent=(
                0,
                self.recording.number_of_cycles,
                len(self.labels.row_labels) - 0.5,
                -0.5,
            ),
        )
        self._set_heading(
            "What each constraint cost the optimizer",
            "darker = more of this constraint was given up",
        )
        self.axes.set_yticks([])
        self.axes.set_xlabel(
            "control cycle", fontsize=LABEL_FONT_SIZE, color=self.palette.muted_ink
        )
        self._style_axes()
        self._add_scale(image, "cost (weight × violation²)")
        self.cursor = self.axes.axvline(
            0,
            color=self.palette.primary_ink,
            linewidth=1.2,
            path_effects=[withStroke(linewidth=3, foreground=self.palette.surface)],
        )
        self._add_task_legend()

    def _add_task_legend(self) -> None:
        """
        Name the color that marks each task, so identity never rests on color alone.
        """
        handles = [
            Patch(facecolor=self.labels.task_color(task_name), label=task_name)
            for task_name in self.labels.ordered_task_names
        ]
        legend = self.axes.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.02),
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=SUBTITLE_FONT_SIZE,
            handlelength=1.0,
            handleheight=1.0,
        )
        for text in legend.get_texts():
            text.set_color(self.palette.secondary_ink)

    @staticmethod
    def _cost_scale(costs: np.ndarray) -> SymLogNorm:
        """
        Build a scale that keeps the cheap constraints visible next to the expensive
        ones.
        """
        largest_cost = float(np.nanmax(costs)) if costs.size else 0.0
        if not largest_cost > 0.0:
            largest_cost = 1.0
        return SymLogNorm(
            linthresh=largest_cost * 10.0**-VISIBLE_COST_DECADES,
            vmin=0.0,
            vmax=largest_cost,
        )

    def draw(self, cycle_index: int) -> None:
        self.cursor.set_xdata([cycle_index, cycle_index])
        self.canvas.draw_idle()


class SensitivityPanel(InspectorPanel):
    """
    Shows how strongly each constraint reacts to each degree of freedom.
    """

    def __init__(self, analysis: ControlCycleAnalysis, labels: ChartLabels) -> None:
        super().__init__(analysis, labels)
        self._create_axes_with_task_strip()
        self.image = self.axes.imshow(
            self.analysis.horizon_sensitivities(0),
            aspect="auto",
            interpolation="nearest",
            cmap=self.palette.polarity_color_map,
            vmin=-1.0,
            vmax=1.0,
        )
        self._set_heading(
            "Which joints each constraint pulls on",
            "the two hues are the two directions a joint can pull it",
        )
        self.axes.set_yticks([])
        degree_of_freedom_labels = self.labels.degree_of_freedom_labels
        self.axes.set_xticks(range(len(degree_of_freedom_labels)))
        self.axes.set_xticklabels(
            degree_of_freedom_labels,
            fontsize=LABEL_FONT_SIZE,
            color=self.palette.secondary_ink,
            rotation=40,
            ha="right",
        )
        self._style_axes()
        self._add_scale(self.image, "change per unit of joint speed")

    def draw(self, cycle_index: int) -> None:
        sensitivities = self.analysis.horizon_sensitivities(cycle_index)
        largest = float(np.max(np.abs(sensitivities))) or 1.0
        self.image.set_data(sensitivities)
        self.image.set_clim(-largest, largest)
        self.canvas.draw_idle()


class ConflictPanel(InspectorPanel):
    """
    Shows which constraints ask for opposite motions.
    """

    def __init__(self, analysis: ControlCycleAnalysis, labels: ChartLabels) -> None:
        super().__init__(analysis, labels)
        self._create_axes_with_task_strip()
        self.image = self.axes.imshow(
            self.analysis.conflict_matrix(0),
            aspect="auto",
            interpolation="nearest",
            cmap=self.palette.polarity_color_map,
            vmin=-1.0,
            vmax=1.0,
        )
        self._set_heading(
            "Which constraints fight each other",
            "warm = the pair wants opposite motions; columns in row order",
        )
        self.axes.set_yticks([])
        number_of_rows = self.recording.structure.number_of_rows
        self.axes.set_xticks(range(number_of_rows))
        self.axes.set_xticklabels(
            range(number_of_rows),
            fontsize=LABEL_FONT_SIZE,
            color=self.palette.muted_ink,
        )
        self._style_axes()
        self._add_scale(self.image, "agreement (-1 opposite, +1 same)")

    def draw(self, cycle_index: int) -> None:
        self.image.set_data(self.analysis.conflict_matrix(cycle_index))
        self.canvas.draw_idle()


class JointLoadPanel(InspectorPanel):
    """
    Shows how much of its velocity limit each degree of freedom spends.
    """

    def __init__(self, analysis: ControlCycleAnalysis, labels: ChartLabels) -> None:
        super().__init__(analysis, labels)
        degree_of_freedom_labels = self.labels.degree_of_freedom_labels
        positions = range(len(degree_of_freedom_labels))
        self.bars = self.axes.barh(
            positions,
            np.zeros(len(degree_of_freedom_labels)),
            color=self.palette.task_colors[0],
            height=0.6,
        )
        self.value_labels = [
            self.axes.text(
                0,
                bar.get_y() + bar.get_height() / 2,
                "",
                fontsize=LABEL_FONT_SIZE,
                color=self.palette.secondary_ink,
                va="center",
            )
            for bar in self.bars
        ]
        self.axes.axvline(1.0, color=self.palette.limit, linewidth=1.2)
        self._set_heading(
            "How hard each joint is working",
            "the red line is the velocity limit",
        )
        self.axes.set_yticks(positions)
        self.axes.set_yticklabels(
            degree_of_freedom_labels,
            fontsize=LABEL_FONT_SIZE,
            color=self.palette.secondary_ink,
        )
        self.axes.set_xticks([0.0, 0.5, 1.0])
        self.axes.set_xticklabels(["0%", "50%", "100%"])
        self.axes.invert_yaxis()
        self.axes.xaxis.grid(True, color=self.palette.gridline, linewidth=1)
        self.axes.set_axisbelow(True)
        self._style_axes()

    def draw(self, cycle_index: int) -> None:
        saturation = self.analysis.velocity_saturation(cycle_index)
        for bar, value_label, value in zip(self.bars, self.value_labels, saturation):
            bar.set_width(value)
            value_label.set_x(value + 0.02)
            value_label.set_text(f"{value:.0%}")
        self.axes.set_xlim(0.0, max(1.15, float(np.max(saturation)) * 1.15))
        self.canvas.draw_idle()


# %% table


class CycleTable(QTableWidget):
    """
    Lists every constraint of the inspected cycle with the numbers behind the charts.

    The charts encode their values as color, which cannot be read off precisely and is
    lost to a colorblind reader; this table is where the same values are exact.
    """

    COLUMN_TITLES = (
        "task",
        "constraint",
        "weight",
        "violation",
        "cost",
        "achieved",
        "asked for",
    )

    def __init__(self, analysis: ControlCycleAnalysis, labels: ChartLabels) -> None:
        super().__init__(analysis.recording.structure.number_of_rows, 7)
        self.analysis = analysis
        self.labels = labels
        self.setHorizontalHeaderLabels(self.COLUMN_TITLES)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    @staticmethod
    def _task_swatch(color: str) -> QIcon:
        """
        A small square of a task's identity color, to sit beside its name in ink.
        """
        pixmap = QPixmap(10, 10)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)

    def show_cycle(self, cycle_index: int) -> None:
        """
        Fill the table with the given cycle, most expensive constraint first.
        """
        recording = self.analysis.recording
        costs = self.analysis.slack_costs[cycle_index]
        violations = self.analysis.bound_violations[cycle_index]
        achieved = self.analysis.achieved_changes[cycle_index]
        order = np.argsort(np.nan_to_num(costs, nan=-1.0))[::-1]
        task_names = self.labels.task_names
        for position, row in enumerate(order):
            asked_for = (
                f"{recording.row_lower_bounds[cycle_index, row]:.4f}"
                if recording.structure.row_is_equality[row]
                else f"{recording.row_lower_bounds[cycle_index, row]:.4f} … "
                f"{recording.row_upper_bounds[cycle_index, row]:.4f}"
            )
            values = (
                task_names[row],
                recording.structure.row_names[row],
                f"{recording.row_weights[cycle_index, row]:.0f}",
                f"{violations[row]:.5f}",
                f"{costs[row]:.3f}",
                f"{achieved[row]:.5f}",
                asked_for,
            )
            for column, value in enumerate(values):
                self.setItem(position, column, QTableWidgetItem(value))
            self.item(position, 0).setIcon(
                self._task_swatch(self.labels.task_color(task_names[row]))
            )
        self.resizeColumnsToContents()
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)


# %% window


class ConstraintInspector(QWidget):
    """
    Window that scrubs through a recorded motion one control cycle at a time.
    """

    def __init__(
        self,
        recording: ControlCycleRecording,
        file_path: str,
        state_publisher: RecordedWorldStatePublisher | None = None,
    ) -> None:
        super().__init__()
        self.state_publisher = state_publisher
        self.analysis = ControlCycleAnalysis(recording)
        self.labels = ChartLabels(recording)
        self.palette_colors = self.labels.palette
        self.timeline = ViolationTimelinePanel(self.analysis, self.labels)
        self.panels: list[InspectorPanel] = [
            self.timeline,
            SensitivityPanel(self.analysis, self.labels),
            ConflictPanel(self.analysis, self.labels),
            JointLoadPanel(self.analysis, self.labels),
        ]
        self.table = CycleTable(self.analysis, self.labels)
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.show_next_cycle)
        self._setup_widgets(file_path)
        self._setup_layout()
        self._setup_shortcuts()
        self.show_cycle(0)

    @property
    def recording(self) -> ControlCycleRecording:
        """
        The recording being inspected.
        """
        return self.analysis.recording

    @property
    def last_cycle_index(self) -> int:
        """
        Index of the final recorded control cycle.
        """
        return self.recording.number_of_cycles - 1

    def _setup_widgets(self, file_path: str) -> None:
        """
        Create the labels, buttons and slider of the window.
        """
        structure = self.recording.structure
        self.header_label = QLabel(
            f"{file_path}  —  {self.recording.number_of_cycles} cycles, "
            f"{structure.number_of_rows} constraints, "
            f"{structure.number_of_degrees_of_freedom} joints"
        )
        self.header_label.setStyleSheet(f"color: {self.palette_colors.secondary_ink};")
        self.cycle_label = QLabel()
        self.cycle_label.setStyleSheet(
            f"color: {self.palette_colors.primary_ink}; font-weight: bold;"
        )
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.last_cycle_index)
        self.slider.valueChanged.connect(self.show_cycle)

        self.first_button = QPushButton("First")
        self.first_button.clicked.connect(self.show_first_cycle)
        self.previous_button = QPushButton("<")
        self.previous_button.clicked.connect(self.show_previous_cycle)
        self.next_button = QPushButton(">")
        self.next_button.clicked.connect(self.show_next_cycle)
        self.last_button = QPushButton("Last")
        self.last_button.clicked.connect(self.show_last_cycle)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)

    def _setup_layout(self) -> None:
        """
        Put the whole motion on top, the inspected cycle below it, and the exact numbers
        at the bottom.
        """
        cycle_charts = QHBoxLayout()
        for panel in self.panels[1:]:
            cycle_charts.addWidget(panel.canvas)

        navigation = QHBoxLayout()
        navigation.addWidget(self.first_button)
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.cycle_label)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.last_button)
        navigation.addWidget(self.play_button)

        layout = QVBoxLayout()
        layout.addWidget(self.header_label)
        layout.addWidget(self.timeline.canvas, stretch=3)
        layout.addWidget(self.slider)
        layout.addLayout(navigation)
        layout.addLayout(cycle_charts, stretch=4)
        layout.addWidget(self.table, stretch=2)
        self.setLayout(layout)
        self.setWindowTitle("Constraint Inspector")
        self.resize(1600, 1000)

    def _setup_shortcuts(self) -> None:
        """
        Bind the navigation keys of the motion statechart inspector.
        """
        for key, action in (
            (Qt.Key_Left, self.show_previous_cycle),
            (Qt.Key_Right, self.show_next_cycle),
            (Qt.Key_Home, self.show_first_cycle),
            (Qt.Key_End, self.show_last_cycle),
            (Qt.Key_Space, self.toggle_playback),
        ):
            QShortcut(QKeySequence(key), self).activated.connect(action)

    def show_cycle(self, cycle_index: int) -> None:
        """
        Show the given control cycle in every panel.
        """
        cycle_index = int(np.clip(cycle_index, 0, self.last_cycle_index))
        self.slider.setValue(cycle_index)
        self.cycle_label.setText(
            f"cycle {cycle_index}/{self.last_cycle_index}  "
            f"t={self.recording.times[cycle_index]:.2f}s"
        )
        for panel in self.panels:
            panel.draw(cycle_index)
        self.table.show_cycle(cycle_index)
        self._publish_world_state(cycle_index)

    def _publish_world_state(self, cycle_index: int) -> None:
        """
        Put the world of everyone on the synchronization topic into the pose of the
        given cycle.
        """
        if self.state_publisher is None:
            return
        self.state_publisher.publish(
            self.recording.world_degree_of_freedom_ids,
            self.recording.world_positions[cycle_index],
        )

    def show_first_cycle(self) -> None:
        """
        Jump to the first recorded cycle.
        """
        self.show_cycle(0)

    def show_last_cycle(self) -> None:
        """
        Jump to the last recorded cycle.
        """
        self.show_cycle(self.last_cycle_index)

    def show_previous_cycle(self) -> None:
        """
        Step one cycle back.
        """
        self.show_cycle(self.slider.value() - 1)

    def show_next_cycle(self) -> None:
        """
        Step one cycle forward, stopping playback at the end of the recording.
        """
        if self.slider.value() >= self.last_cycle_index:
            self.stop_playback()
            return
        self.show_cycle(self.slider.value() + 1)

    def toggle_playback(self) -> None:
        """
        Start playing the recording, or stop it if it is already playing.
        """
        if self.playback_timer.isActive():
            self.stop_playback()
            return
        if self.slider.value() >= self.last_cycle_index:
            self.show_first_cycle()
        self.play_button.setText("Pause")
        self.playback_timer.start(PLAYBACK_INTERVAL_IN_MILLISECONDS)

    def stop_playback(self) -> None:
        """
        Stop playing the recording.
        """
        self.playback_timer.stop()
        self.play_button.setText("Play")


def _create_state_publisher() -> RecordedWorldStatePublisher:
    """
    Start a ros node that replays the recorded world state.

    Imported here rather than at module level so the viewer opens a recording on a
    machine without ros installed.
    """
    import rclpy

    from giskardpy.qp.recorded_state_publisher import (
        REPLAY_NODE_NAME,
        RecordedWorldStatePublisher,
    )

    rclpy.init()
    return RecordedWorldStatePublisher.for_node(rclpy.create_node(REPLAY_NODE_NAME))


def main() -> None:
    """
    Open a recording named on the command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="path of a recorded motion")
    parser.add_argument(
        "--replay-world-state",
        action="store_true",
        help="publish the world state of the inspected cycle, so a running standalone "
        "giskard shows the robot where it was; never use this against a real robot",
    )
    arguments = parser.parse_args()

    application = QApplication(sys.argv)
    inspector = ConstraintInspector(
        ControlCycleRecording.load(arguments.recording),
        arguments.recording,
        state_publisher=(
            _create_state_publisher() if arguments.replay_world_state else None
        ),
    )
    inspector.show()
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
