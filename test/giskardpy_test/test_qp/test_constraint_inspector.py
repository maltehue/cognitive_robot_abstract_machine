import pytest

from .test_control_cycle_analysis import build_opposing_constraints_recording

pytest.importorskip("PyQt5", reason="the constraint inspector is a PyQt5 tool")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from giskardpy.qp.constraint_inspector import (  # noqa: E402
    ChartLabels,
    ChartPalette,
    ConstraintInspector,
)

OFFSCREEN_ARGUMENTS = ["test", "-platform", "offscreen"]
"""
Lets the window render without a display, so the viewer can be tested anywhere.
"""

LAST_CYCLE_INDEX = 1


@pytest.fixture(scope="module")
def application() -> QApplication:
    existing_application = QApplication.instance()
    if existing_application is not None:
        return existing_application
    return QApplication(OFFSCREEN_ARGUMENTS)


@pytest.fixture()
def inspector(application) -> ConstraintInspector:
    return ConstraintInspector(build_opposing_constraints_recording(), "recording.npz")


# %% navigation


def test_inspector_opens_on_the_first_cycle(inspector):
    assert inspector.slider.value() == 0


def test_slider_covers_every_recorded_cycle(inspector):
    assert inspector.slider.minimum() == 0
    assert inspector.slider.maximum() == LAST_CYCLE_INDEX


def test_stepping_forward_moves_to_the_next_cycle(inspector):
    inspector.show_next_cycle()

    assert inspector.slider.value() == 1


def test_stepping_back_from_the_first_cycle_stays_there(inspector):
    inspector.show_previous_cycle()

    assert inspector.slider.value() == 0


def test_stepping_past_the_last_cycle_stays_there(inspector):
    inspector.show_last_cycle()
    inspector.show_next_cycle()

    assert inspector.slider.value() == LAST_CYCLE_INDEX


def test_dragging_the_slider_selects_the_cycle(inspector):
    inspector.slider.setValue(LAST_CYCLE_INDEX)

    assert inspector.cycle_label.text().startswith(f"cycle {LAST_CYCLE_INDEX}/")


def test_the_selected_time_is_shown(inspector):
    recording = build_opposing_constraints_recording()
    inspector.show_last_cycle()

    assert f"t={recording.times[LAST_CYCLE_INDEX]:.2f}s" in inspector.cycle_label.text()


# %% panels


def test_every_panel_marks_the_selected_cycle(inspector):
    inspector.show_last_cycle()

    assert list(inspector.timeline.cursor.get_xdata()) == [
        LAST_CYCLE_INDEX,
        LAST_CYCLE_INDEX,
    ]


def test_redrawing_a_panel_does_not_add_another_color_scale(inspector):
    """
    A panel that rebuilt its artists on every cycle stacked up one color scale per
    redraw until the chart itself was squeezed out of the figure.
    """
    scales_per_panel = [len(panel.figure.axes) for panel in inspector.panels]

    inspector.show_last_cycle()
    inspector.show_first_cycle()

    assert [len(panel.figure.axes) for panel in inspector.panels] == scales_per_panel


# %% reading the values


def test_the_table_lists_the_most_expensive_constraint_first(inspector):
    inspector.show_first_cycle()

    assert inspector.table.item(0, 1).text() == "push#1/0"


def test_the_table_names_constraints_in_full(inspector):
    """
    The charts label rows with a shortened name, so the table is where the name a motion
    statechart actually generated can be read.
    """
    listed_names = {
        inspector.table.item(row, 1).text() for row in range(inspector.table.rowCount())
    }

    assert listed_names == {"pull#0/0", "push#1/0"}


# %% labels


def _labels_for(
    row_names: list[str], degree_of_freedom_names: list[str]
) -> ChartLabels:
    recording = build_opposing_constraints_recording()
    recording.structure.row_names = row_names
    recording.structure.degree_of_freedom_names = degree_of_freedom_names
    return ChartLabels(recording)


def test_rows_of_the_same_task_are_numbered(inspector):
    labels = _labels_for(["Align#1/0", "Align#1/1", "Align#1/2"], ["a/b_joint"])

    assert labels.row_labels == ["Align 0", "Align 1", "Align 2"]


def test_a_task_owning_one_row_is_not_numbered(inspector):
    labels = _labels_for(["Align#1/0", "Fill#2/0"], ["a/b_joint"])

    assert labels.row_labels == ["Align", "Fill"]


def test_an_over_long_row_label_keeps_both_of_its_ends(inspector):
    labels = _labels_for(["KeepSourceRimAboveReceiverRim#4/clearance"], ["a/b_joint"])

    assert labels.row_labels == ["KeepSourceRi…veReceiverRim"]


def test_joint_labels_drop_the_robot_prefix_and_joint_suffix(inspector):
    labels = _labels_for(["Align#1/0"], ["tracy/left_wrist_1_joint"])

    assert labels.degree_of_freedom_labels == ["left_wrist_1"]


def test_tasks_take_the_identity_colors_in_a_fixed_order(inspector):
    labels = _labels_for(["Second#1/0", "First#2/0"], ["a/b_joint"])
    palette = ChartPalette()

    assert labels.task_color("Second") == palette.task_colors[0]
    assert labels.task_color("First") == palette.task_colors[1]


def test_tasks_past_the_palette_share_one_color(inspector):
    row_names = [f"Task{index}#{index}/0" for index in range(10)]
    labels = _labels_for(row_names, ["a/b_joint"])
    palette = ChartPalette()

    assert labels.task_color("Task8") == palette.overflow_task_color
    assert labels.task_color("Task9") == palette.overflow_task_color


def test_playback_runs_until_the_end_of_the_recording(inspector):
    inspector.toggle_playback()
    assert inspector.playback_timer.isActive()

    inspector.show_last_cycle()
    inspector.show_next_cycle()

    assert not inspector.playback_timer.isActive()
