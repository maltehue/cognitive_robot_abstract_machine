"""
What a demo pour was asked to do, kept in its bag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.real_pouring.pour_settings import (
    POUR_SETTINGS_FILE_NAME,
    MissingPourSettingsError,
    PourSettings,
)

# %% settings written into a bag and read back


@pytest.fixture
def settings() -> PourSettings:
    """
    A coupled pour to a fill of 0.4.
    """
    return PourSettings(
        goal_fill=0.4,
        fill_tolerance=0.01,
        couple_source_fill=True,
        prediction_duration=1.5,
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
