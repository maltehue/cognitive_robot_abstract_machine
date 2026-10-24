"""
What a demo pour was asked to do, kept in its bag.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.pour_settings import (
    POUR_SETTINGS_FILE_NAME,
    MissingPourSettingsError,
    PourLabel,
    PourLabelMarker,
    PourSettings,
    RepositoryState,
    TiltPacingSettings,
)
from experiments.real_pouring.receiving_cup import ReceivingCup

# %% settings written into a bag and read back


@pytest.fixture
def settings() -> PourSettings:
    """
    A coupled pour of corn from a cup filled to 0.7 to a fill of 0.4, with every setting
    recorded.
    """
    return PourSettings(
        goal_fill=0.4,
        fill_tolerance=0.01,
        couple_source_fill=True,
        prediction_duration=1.5,
        start_fill=0.7,
        material=PouredMaterial.CORN,
        receiving_cup=ReceivingCup.TRANSLUCENT_CUP,
        pouring_velocity=0.03,
        stopping_velocity=1.0,
        settled_inflow=0.005,
        tilt_pacing=TiltPacingSettings(
            approach_rate=math.radians(1.5),
            maximum_rate=math.radians(3.0),
            maximum_return_rate=math.radians(10.0),
            maximum_acceleration=math.radians(30.0),
            maximum_tilt=math.radians(110.0),
            ceiling_slowdown_width=math.radians(5.0),
            transfer_start_tilt=math.radians(40.0),
            ceiling_hold_seconds=5.0,
        ),
        drain=ModelPoint(
            repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
        ),
        bank=BankSettings(tracks_onset=True, trickle=0.0),
        stop_lead_time=1.0,
        corrects_drain=False,
        learned_head=False,
        repository=RepositoryState(commit="0" * 40, has_uncommitted_changes=True),
    )


def test_settings_read_back_from_the_bag(
    settings: PourSettings, tmp_path: Path
) -> None:
    """
    The settings a bag was recorded with come back unchanged.
    """
    (tmp_path / POUR_SETTINGS_FILE_NAME).write_text(settings.as_json_text())
    assert PourSettings.read(tmp_path) == settings


def test_a_bag_recorded_without_settings_says_so(tmp_path: Path) -> None:
    """
    Bags from before the settings were recorded have no settings to read.
    """
    with pytest.raises(MissingPourSettingsError):
        PourSettings.read(tmp_path)


def test_settings_recorded_before_the_later_fields_still_read(
    tmp_path: Path,
) -> None:
    """
    A bag of 2026-10-23 recorded only the goal, the tolerance, the coupling and the
    prediction window; it reads with those and with nothing for what came later.
    """
    recorded = Path(__file__).parent / "dataset" / "pour_settings"
    original = recorded / "2026-10-23_17-09-13.json"
    (tmp_path / POUR_SETTINGS_FILE_NAME).write_text(original.read_text())
    fields = {
        key: value
        for key, value in json.loads(original.read_text()).items()
        if not key.startswith("__")
    }
    assert PourSettings.read(tmp_path) == PourSettings(**fields)


# %% the code a pour ran with


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """
    A git repository with one committed file.
    """
    run = lambda *arguments: subprocess.run(
        ["git", "-C", str(tmp_path), *arguments], check=True, capture_output=True
    )
    run("init")
    (tmp_path / "file.txt").write_text("first")
    run("add", "file.txt")
    run(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "first",
    )
    return tmp_path


def test_the_commit_checked_out_is_recorded(repository: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = RepositoryState.of_working_tree(repository)
    assert state.commit == head
    assert not state.has_uncommitted_changes


def test_uncommitted_changes_are_noticed(repository: Path) -> None:
    (repository / "file.txt").write_text("changed")
    assert RepositoryState.of_working_tree(repository).has_uncommitted_changes


# %% the label a pour's bag is named with


def test_the_label_names_material_cups_goal_start_and_approach() -> None:
    label = PourLabel(
        material=PouredMaterial.CORN,
        receiving_cup=ReceivingCup.TRANSLUCENT_CUP,
        goal_fill=0.3,
        start_fill=1.0,
        approach_tilt_rate=math.radians(3.0),
        live_bank=True,
        learned_head=False,
    )

    assert label.text() == (
        f"{PouredMaterial.CORN}_{ReceivingCup.TRANSLUCENT_CUP}_goal30_start100_approach3"
    )


def test_the_label_marks_a_fractional_rate_and_departures_from_the_usual_setup() -> (
    None
):
    """
    A run without the live bank or with the learned head says so; a fractional rate
    keeps its decimal without a dot, which a ROS name would refuse.
    """
    label = PourLabel(
        material=PouredMaterial.CORN,
        receiving_cup=ReceivingCup.TRANSLUCENT_CUP,
        goal_fill=0.2,
        start_fill=0.7,
        approach_tilt_rate=math.radians(1.5),
        live_bank=False,
        learned_head=True,
    )

    assert label.text() == (
        f"{PouredMaterial.CORN}_{ReceivingCup.TRANSLUCENT_CUP}_goal20_start70_approach1p5"
        f"_{PourLabelMarker.WITHOUT_BANK}_{PourLabelMarker.WITH_LEARNED_HEAD}"
    )


def test_the_label_marks_a_bank_that_tells_only_the_fills() -> None:
    label = PourLabel(
        material=PouredMaterial.CORN,
        receiving_cup=ReceivingCup.TRANSLUCENT_CUP,
        goal_fill=0.3,
        start_fill=1.0,
        approach_tilt_rate=math.radians(3.0),
        live_bank=True,
        learned_head=False,
        corrects_drain=False,
    )

    assert label.text() == (
        f"{PouredMaterial.CORN}_{ReceivingCup.TRANSLUCENT_CUP}_goal30_start100_approach3"
        f"_{PourLabelMarker.FILLS_ONLY}"
    )
