"""Non-Boosted Dynamic helicopter spring mode: rotor-q gradient, disc-attitude
bias force with q-based fallback, motorized-trim anchor slew, friction and
n-per-rev vibration lifecycles, and the pedal branch."""
import time

import pytest

from tests.framework.base import BaseTelemetryEffectTestCase
from tests.framework.utils import TelemetryDataBuilder
from telemffb.sim.msfs_xp.Helicopter import Helicopter
from telemffb.SettingsManager import SpringModeEnum
import telemffb.globals as G

# Full-suite runs set this via earlier tests; standalone runs need it present
# (framework supports_axis_override reads it unconditionally).
G.device_firmware_version = getattr(G, "device_firmware_version", None)

# The Non-Boosted mode lives on the mech-cyclic branch; skip cleanly when this
# file is present in a working tree whose branch doesn't carry the feature.
if not hasattr(SpringModeEnum, "NONBOOSTED"):
    pytestmark = pytest.mark.skip(reason="SpringModeEnum.NONBOOSTED not on this branch")


@pytest.mark.unit
@pytest.mark.msfs
@pytest.mark.helicopter
@pytest.mark.joystick
class TestNonBoostedCyclic(BaseTelemetryEffectTestCase):

    def _make(self, **telem_fields):
        instance = self.create_aircraft_instance(
            Helicopter, name="TestHeli", _test_sim_is_msfs=True,
            _test_device_type="joystick")
        instance._test_sim_is_msfs = True
        instance._test_device_type = "joystick"
        instance.spring_mode = SpringModeEnum.NONBOOSTED
        instance.cyclic_spring_init = 1
        builder = (TelemetryDataBuilder()
                   .on_ground(False)
                   .with_airspeed(0.0)
                   .with_field("AircraftClass", "Helicopter")
                   .with_field("FFBType", "joystick")
                   .with_field("RotorRPMPct", 1.0))
        for k, v in telem_fields.items():
            builder = builder.with_field(k, v)
        telem = builder.build()
        self.set_telemetry(instance, telem)
        self.mock_device._input_data.set_axis(x=0.0, y=0.0)
        return instance, telem

    def test_gradient_scales_with_rotor_rpm_squared(self):
        # full RRPM at zero airspeed: the stick is stiff in the hover
        inst, telem = self._make()
        inst.msfs_update_heli_controls(telem)
        full = inst.spring_x.positiveCoefficient
        assert full == pytest.approx(4096 * inst.nb_cyclic_gradient, abs=2)
        assert inst.spring_y.positiveCoefficient == full

        # half RRPM: gradient falls with the square
        inst2, telem2 = self._make(RotorRPMPct=0.5)
        inst2.msfs_update_heli_controls(telem2)
        assert inst2.spring_x.positiveCoefficient == pytest.approx(full * 0.25, abs=2)

    def test_gradient_gets_forward_speed_boost(self):
        inst, telem = self._make()
        telem["IAS"] = inst.NB_VREF_MS  # q_frac = 1.0
        inst.msfs_update_heli_controls(telem)
        boosted = inst.spring_x.positiveCoefficient
        expect = 4096 * inst.nb_cyclic_gradient * (1 + inst.NB_GRAD_Q_BOOST)
        assert boosted == pytest.approx(expect, abs=2)

    def test_bias_zero_on_ground_regardless_of_disc_and_stick(self):
        # THE runaway regression: at zero airspeed the bias must be exactly
        # zero no matter what the disc angles report or where the stick is —
        # only the gradient and friction act, so the stick cannot run away.
        inst, telem = self._make(DiskBank=6.0, DiskPitch=4.0)
        self.mock_device._input_data.set_axis(x=0.4, y=0.2)
        inst.msfs_update_heli_controls(telem)
        assert telem["_nb_bias"] == [0.0, 0.0]
        assert abs(self.mock_effects["nb_cyclic_bias"]._magnitude) == pytest.approx(0.0, abs=1e-9)

    def test_bias_independent_of_stick_position(self):
        # Structural anti-feedback property: same flight state, different
        # stick positions -> identical bias.
        inst, telem = self._make()
        telem["IAS"] = inst.NB_VREF_MS / (2 ** 0.5)
        self.mock_device._input_data.set_axis(x=0.0, y=0.0)
        inst.msfs_update_heli_controls(telem)
        centered = list(telem["_nb_bias"])
        inst2, telem2 = self._make()
        telem2["IAS"] = inst.NB_VREF_MS / (2 ** 0.5)
        self.mock_device._input_data.set_axis(x=0.6, y=-0.5)
        inst2.msfs_update_heli_controls(telem2)
        assert telem2["_nb_bias"] == pytest.approx(centered, abs=1e-9)

    def test_bias_grows_with_forward_q(self):
        # q_frac = 0.5 at IAS = NB_VREF/sqrt(2), full RRPM
        inst, telem = self._make()
        telem["IAS"] = inst.NB_VREF_MS / (2 ** 0.5)
        inst.msfs_update_heli_controls(telem)
        bias_pitch, bias_roll = telem["_nb_bias"]
        assert bias_pitch == pytest.approx(inst.nb_bias_gain_pitch * 0.5, abs=1e-6)
        assert bias_roll == pytest.approx(inst.nb_bias_gain_roll * 0.5, abs=1e-6)

    def test_bias_scales_with_rotor_q(self):
        # same airspeed at half RRPM -> quarter the bias (spooling down kills it)
        inst, telem = self._make(RotorRPMPct=0.5)
        telem["IAS"] = inst.NB_VREF_MS / (2 ** 0.5)
        inst.msfs_update_heli_controls(telem)
        bias_pitch, _ = telem["_nb_bias"]
        assert bias_pitch == pytest.approx(inst.nb_bias_gain_pitch * 0.5 * 0.25, abs=1e-6)

    def test_lateral_sign_follows_rotor_direction(self):
        inst, telem = self._make()
        telem["IAS"] = inst.NB_VREF_MS
        inst.nb_rotor_direction = "Clockwise"
        inst.msfs_update_heli_controls(telem)
        _, bias_roll = telem["_nb_bias"]
        assert bias_roll < 0  # flipped relative to the Counter-Clockwise default

    def test_motorized_trim_slews_anchor_rate_limited(self):
        inst, telem = self._make(CyclicTrimY=0.5, CyclicTrimX=0.0)
        inst.nb_motorized_trim = True
        # first frame initializes the slew clock (dt == 0, no movement)
        inst.msfs_update_heli_controls(telem)
        assert inst.cpO_y == 0
        # backdate the clock 0.1s: anchor moves nb_trim_rate*4096*0.1 ~ 102
        inst._nb_last_slew_t = time.perf_counter() - 0.1
        inst.msfs_update_heli_controls(telem)
        assert 0 < inst.cpO_y <= inst.nb_trim_rate * 4096 * 0.11
        assert inst.cpO_y < 2048  # rate-limited, not an instant recapture

    def test_anchor_fixed_without_motorized_trim(self):
        inst, telem = self._make(CyclicTrimY=0.5)
        inst.nb_motorized_trim = False
        inst._nb_last_slew_t = time.perf_counter() - 1.0
        inst.msfs_update_heli_controls(telem)
        assert inst.cpO_y == 0

    def test_friction_and_bias_destroyed_on_mode_switch(self):
        inst, telem = self._make()
        inst.msfs_update_heli_controls(telem)
        fric = self.mock_effects["nb_friction"]
        assert fric.started
        assert fric._x_coefficient == int(4096 * inst.nb_friction)
        assert self.mock_effects["nb_cyclic_bias"].started

        inst.spring_mode = SpringModeEnum.NOSPRING
        inst.msfs_update_heli_controls(telem)
        assert not self.mock_effects["nb_friction"].started
        assert not self.mock_effects["nb_cyclic_bias"].started
        # NOSPRING behavior unchanged: coefficients zeroed
        assert inst.spring_x.positiveCoefficient == 0

    def test_nrev_vibration_frequency_and_load_scaling(self):
        inst, telem = self._make(RotorRPM=300.0)
        inst.rotor_blade_count = 2
        telem["G"] = 1.0
        inst.ac_update_nonboosted_vibration(telem)
        eff = self.mock_effects["nb_nrev_y"]
        assert eff.started
        freq, mag, direction, _ = eff._periodic
        assert freq == pytest.approx(300.0 / 60.0 * 2)  # 10 Hz two-blade
        assert mag == pytest.approx(inst.nb_vibration_intensity * 0.3, abs=1e-6)
        assert self.mock_effects["nb_nrev_x"]._periodic[2] == 90

        # blade slap (XP) adds directly to the magnitude
        telem["BladeSlap"] = 0.5
        inst.ac_update_nonboosted_vibration(telem)
        _, mag_slap, _, _ = self.mock_effects["nb_nrev_y"]._periodic
        assert mag_slap == pytest.approx(
            inst.nb_vibration_intensity * (0.3 + 0.5), abs=1e-6)

    def test_vibration_disposed_when_mode_inactive(self):
        inst, telem = self._make(RotorRPM=300.0)
        inst.ac_update_nonboosted_vibration(telem)
        assert self.mock_effects["nb_nrev_y"].started
        inst.spring_mode = SpringModeEnum.NOSPRING
        inst.ac_update_nonboosted_vibration(telem)
        assert not self.mock_effects["nb_nrev_y"].started


@pytest.mark.unit
@pytest.mark.msfs
@pytest.mark.helicopter
@pytest.mark.pedals
class TestNonBoostedPedals(BaseTelemetryEffectTestCase):

    def _make(self, **telem_fields):
        instance = self.create_aircraft_instance(
            Helicopter, name="TestHeli", _test_sim_is_msfs=True,
            _test_device_type="pedals")
        instance._test_sim_is_msfs = True
        instance._test_device_type = "pedals"
        instance.spring_mode = SpringModeEnum.NONBOOSTED
        instance.telemffb_controls_axes = True
        instance.pedals_init = 1
        builder = (TelemetryDataBuilder()
                   .on_ground(False)
                   .with_airspeed(0.0)
                   .with_field("AircraftClass", "Helicopter")
                   .with_field("FFBType", "pedals")
                   .with_field("RotorRPMPct", 1.0))
        for k, v in telem_fields.items():
            builder = builder.with_field(k, v)
        telem = builder.build()
        self.set_telemetry(instance, telem)
        self.mock_device._input_data.set_axis(x=0.0)
        return instance, telem

    def test_pedal_gradient_and_tail_pitch_bias(self):
        inst, telem = self._make(TailRotorPedalPos=0.5)
        inst.msfs_update_pedals(telem)
        assert inst.spring_x.positiveCoefficient == pytest.approx(
            4096 * inst.nb_pedal_gradient, abs=2)
        eff = self.mock_effects["nb_pedal_bias"]
        assert eff.started
        # bias loads AGAINST the applied pedal (NB_PEDAL_SIGN) — same-direction
        # would be positive feedback
        assert eff._magnitude == pytest.approx(
            inst.nb_pedal_bias_gain * 0.5 * inst.NB_PEDAL_SIGN, abs=1e-6)

    def test_pedal_bias_destroyed_when_leaving_mode(self):
        inst, telem = self._make(TailRotorPedalPos=0.5)
        inst.msfs_update_pedals(telem)
        assert self.mock_effects["nb_pedal_bias"].started
        inst.spring_mode = SpringModeEnum.NOSPRING
        inst.msfs_update_pedals(telem)
        assert not self.mock_effects["nb_pedal_bias"].started
        assert inst.spring_x.positiveCoefficient == 0
