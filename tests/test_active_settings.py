"""mark_settings_active() and the main window's merge of those settings with
the started effects' settings."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

import telemffb.globals as G
import telemffb.utils as utils
from telemffb.MainWindow import MainWindow
from telemffb.sim.BaseTelemetryData import BaseTelemetryData
from telemffb.sim.base.AircraftEffectUtilsBase import AircraftEffectUtilsBase
from telemffb.telem.telem_format import ACTIVE_SETTINGS_KEY

pytestmark = pytest.mark.unit


def _aircraft_with_frame():
    aircraft = AircraftEffectUtilsBase()
    aircraft._telem_data = BaseTelemetryData()
    return aircraft


class TestMarkSettingsActive:
    def test_nothing_marked_leaves_the_frame_alone(self):
        aircraft = _aircraft_with_frame()
        aircraft.mark_settings_active()
        assert aircraft.telem_data[ACTIVE_SETTINGS_KEY] is None

    def test_names_accumulate_for_the_frame_without_repeats(self):
        aircraft = _aircraft_with_frame()
        aircraft.mark_settings_active("ffb_api_fly_through_deadzone")
        aircraft.mark_settings_active("ffb_api_fly_through_release_deadzone",
                                      "ffb_api_fly_through_deadzone")
        assert aircraft.telem_data[ACTIVE_SETTINGS_KEY] == [
            "ffb_api_fly_through_deadzone", "ffb_api_fly_through_release_deadzone"]

    def test_a_new_frame_starts_empty(self):
        aircraft = _aircraft_with_frame()
        aircraft.mark_settings_active("ffb_api_fly_through_deadzone")
        aircraft._telem_data = BaseTelemetryData()
        assert aircraft.telem_data[ACTIVE_SETTINGS_KEY] is None


class TestOwnActiveSettings:
    """MainWindow._own_active_effects(), called unbound with a stub window."""

    @pytest.fixture
    def window(self):
        return SimpleNamespace(_configured_value=lambda name: None,
                               _slider_factor=lambda name: None)

    @pytest.fixture
    def started_effect(self, monkeypatch):
        effect = SimpleNamespace(started=True, name="rotor_rumble", id=1, intensity=0.5,
                                 type_id=0, axis_gains=None)
        monkeypatch.setattr(G, "effects", SimpleNamespace(dict={"rotor_rumble": effect}), raising=False)
        monkeypatch.setattr(utils.EffectTranslator, "get_translation",
                            staticmethod(lambda name: ("Rotor Rumble", "rotor_rumble_intensity")))

    def test_frame_settings_join_the_started_effects_settings(self, window, started_effect):
        frame = {ACTIVE_SETTINGS_KEY: ["ffb_api_fly_through_release_deadzone"]}
        _, settings = MainWindow._own_active_effects(window, frame)
        assert settings == ["rotor_rumble_intensity", "ffb_api_fly_through_release_deadzone"]

    def test_a_setting_both_sources_name_is_listed_once(self, window, started_effect):
        frame = {ACTIVE_SETTINGS_KEY: ["rotor_rumble_intensity"]}
        _, settings = MainWindow._own_active_effects(window, frame)
        assert settings == ["rotor_rumble_intensity"]

    def test_without_a_frame_only_the_started_effects_count(self, window, started_effect):
        _, settings = MainWindow._own_active_effects(window)
        assert settings == ["rotor_rumble_intensity"]
