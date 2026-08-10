"""Next Gen Dynamic spring: the soft-knee force curve, the cruise-anchor
derivation (telemffb.utils pure functions), and the NEXTGEN branch of the
MSFS/X-Plane flight-controls mixin."""
import pytest

from tests.framework.base import BaseTelemetryEffectTestCase
from tests.framework.utils import TelemetryDataBuilder
from telemffb.sim.msfs_xp.MsfsXpFlightControlsMixIn import MsfsXpFlightControlsMixIn
from telemffb.SettingsManager import SpringModeEnum
from telemffb.util.conversions import ms2kt
from telemffb.utils import nextgen_spring_curve, nextgen_anchor_ratio

pytestmark = [pytest.mark.unit, pytest.mark.msfs, pytest.mark.joystick]

# C172 numbers used throughout the design discussion: VC 63 m/s against an
# 84 m/s red-line -> anchor ratio (63/84)^2
A_C172 = (63.0 / 84.0) ** 2


class _Telem:
    """Attribute bag standing in for BaseTelemetryData."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestCurve:
    def test_anchor_and_vne_endpoints(self):
        # exactly at the anchor the curve hits the cruise-force plateau,
        # at Vne it always reaches full device force
        assert nextgen_spring_curve(A_C172, A_C172, 0.7) == pytest.approx(0.8)
        assert nextgen_spring_curve(1.0, A_C172, 0.7) == pytest.approx(1.0)
        assert nextgen_spring_curve(0.0, A_C172, 0.7) == pytest.approx(0.0)

    def test_c172_worked_numbers(self):
        # approach (~65 kt, c=0.16) lifts from 16% raw to ~33% device
        out = nextgen_spring_curve(0.16, A_C172, 0.7)
        assert out == pytest.approx(0.332, abs=0.005)
        # shallow overspeed dive (c=0.8) sits between cruise and Vne force
        dive = nextgen_spring_curve(0.8, A_C172, 0.7)
        assert 0.8 < dive < 1.0
        assert dive == pytest.approx(
            0.8 + 0.2 * (0.8 - A_C172) / (1.0 - A_C172), abs=1e-9)

    def test_monotonic_and_continuous_across_knee(self):
        a = A_C172
        grid = [i / 200.0 for i in range(201)]
        outs = [nextgen_spring_curve(c, a, 0.7) for c in grid]
        assert all(y2 >= y1 for y1, y2 in zip(outs, outs[1:]))
        # no jump at the knee itself
        below = nextgen_spring_curve(a - 1e-9, a, 0.7)
        above = nextgen_spring_curve(a + 1e-9, a, 0.7)
        assert above - below == pytest.approx(0.0, abs=1e-6)

    def test_floor_clamps_low_end_only(self):
        assert nextgen_spring_curve(0.0, A_C172, 0.7, floor=0.06) == 0.06
        # floor is a max(), not an offset: mid-envelope output unaffected
        assert nextgen_spring_curve(0.3, A_C172, 0.7, floor=0.06) == \
            nextgen_spring_curve(0.3, A_C172, 0.7, floor=0.0)

    def test_gamma_one_anchor_one_cf_one_degenerates_to_identity(self):
        # Basic-equivalence sanity: with no compression, no plateau and the
        # anchor pushed to Vne, the curve is (numerically) the identity.
        for c in (0.05, 0.16, 0.4, 0.59, 0.85, 1.0):
            out = nextgen_spring_curve(c, 1.0, 1.0, cruise_force=1.0)
            assert out == pytest.approx(c, abs=2e-3)

    def test_inputs_clamped_to_unit_range(self):
        assert nextgen_spring_curve(1.5, A_C172, 0.7) == 1.0
        assert nextgen_spring_curve(-0.2, A_C172, 0.7) == 0.0

    def test_lower_gamma_lifts_low_speed_more(self):
        soft = nextgen_spring_curve(0.16, A_C172, 0.5)
        stiff = nextgen_spring_curve(0.16, A_C172, 1.0)
        assert soft > stiff


class TestAnchorRatio:
    def test_msfs_design_cruise(self):
        td = _Telem(DesignSpeed=[63.0, 24.0, 25.0], RefMaxIAS=84.0)
        assert nextgen_anchor_ratio(td, "MSFS", 84.0) == \
            pytest.approx((63.0 / 84.0) ** 2)

    def test_msfs_redline_caps_cruise(self):
        td = _Telem(DesignSpeed=[80.0, 24.0, 25.0], RefMaxIAS=84.0)
        assert nextgen_anchor_ratio(td, "MSFS", 84.0) == \
            pytest.approx(0.85 ** 2)

    def test_msfs_vne_fallback_when_no_refmax(self):
        td = _Telem(DesignSpeed=[63.0, 24.0, 25.0], RefMaxIAS=None)
        assert nextgen_anchor_ratio(td, "MSFS", 84.0) == \
            pytest.approx((63.0 / 84.0) ** 2)

    def test_xplane_vno(self):
        td = _Telem(Vno=70.0)
        assert nextgen_anchor_ratio(td, "XPLANE", 90.0) == \
            pytest.approx((70.0 / 90.0) ** 2)

    def test_xplane_vno_capped_at_085_vne(self):
        td = _Telem(Vno=88.0)
        assert nextgen_anchor_ratio(td, "XPLANE", 90.0) == \
            pytest.approx(0.85 ** 2)

    def test_override_wins_over_telemetry(self):
        td = _Telem(DesignSpeed=[63.0, 24.0, 25.0], RefMaxIAS=84.0)
        assert nextgen_anchor_ratio(td, "MSFS", 84.0, override_ms=50.0) == \
            pytest.approx((50.0 / 84.0) ** 2)

    def test_missing_data_falls_back_to_075_vne(self):
        assert nextgen_anchor_ratio(_Telem(), "MSFS", 84.0) == \
            pytest.approx(0.75 ** 2)
        assert nextgen_anchor_ratio(_Telem(), "XPLANE", 90.0) == \
            pytest.approx(0.75 ** 2)
        assert nextgen_anchor_ratio(_Telem(), "DCS", 90.0) == \
            pytest.approx(0.75 ** 2)

    def test_anchor_never_exceeds_095_vne(self):
        # an override at/above Vne must still leave a dive band
        td = _Telem()
        assert nextgen_anchor_ratio(td, "XPLANE", 90.0, override_ms=200.0) == \
            pytest.approx(0.95 ** 2)

    def test_unusable_vne_returns_full_ratio(self):
        assert nextgen_anchor_ratio(_Telem(), "MSFS", 0.0) == 1.0
        assert nextgen_anchor_ratio(_Telem(), "MSFS", None) == 1.0


class TestNextGenMixin(BaseTelemetryEffectTestCase):
    """NEXTGEN branch of MsfsXpFlightControlsMixIn against the pure curve."""

    def _make(self, mode, ffb_type="joystick"):
        instance = self.create_test_instance(MsfsXpFlightControlsMixIn)
        instance._test_sim_is_msfs = True
        instance.spring_mode = mode
        telem = (
            TelemetryDataBuilder()
            .set("src", "MSFS")
            .set("IAS", 50.0)
            .set("DesignSpeed", (100.0, 50.0, 60.0))
            .set("DynPressure", 1000.0)
            .set("AirDensity", 1.225)
            .set("PropThrust", 0.0)
            .set("Incidence", [0.1, -0.05, 1.0])
            .set("AccBody", [0, 1, 0])
            .set("RudderDefl", 0.0)
            .set("G", 1.0)
            .ffb_type(ffb_type)
            .build()
        )
        return instance, telem

    def test_coefficients_match_pure_curve_applied_to_basic_raw(self):
        # BASIC with expo 0 exposes the raw normalized coefficients; the
        # NEXTGEN branch must equal the pure curve applied to those — each
        # axis through its own force-response gamma.
        basic, telem_b = self._make(SpringModeEnum.BASIC)
        basic.on_telemetry(telem_b)

        ng, telem_n = self._make(SpringModeEnum.NEXTGEN)
        ng.nextgen_elevator_response = 0.6
        ng.nextgen_aileron_response = 0.8
        ng.nextgen_rudder_response = 1.0
        ng.on_telemetry(telem_n)

        vne = telem_n["Vne_kt"] / ms2kt
        a = nextgen_anchor_ratio(telem_n, "MSFS", vne)
        for key, gamma in (("_elev_coeff", 0.6), ("_aile_coeff", 0.8),
                           ("_rud_coeff", 1.0)):
            expect = nextgen_spring_curve(
                telem_b[key], a, gamma,
                ng.NEXTGEN_CRUISE_FORCE, ng.nextgen_min_force)
            assert telem_n[key] == pytest.approx(expect, abs=1e-9), key

    def test_floor_holds_at_standstill(self):
        ng, telem = self._make(SpringModeEnum.NEXTGEN)
        telem["IAS"] = 0.0
        telem["DynPressure"] = 0.0
        ng.on_telemetry(telem)
        assert telem["_elev_coeff"] == pytest.approx(ng.nextgen_min_force)
        assert telem["_aile_coeff"] == pytest.approx(ng.nextgen_min_force)

    def test_damper_started_in_nextgen_and_destroyed_on_mode_switch(self):
        ng, telem = self._make(SpringModeEnum.NEXTGEN)
        ng.on_telemetry(telem)
        eff = ng.effects["nextgen_damper"]
        assert eff.started
        assert eff._x_coefficient == int(4096 * ng.nextgen_damper)
        assert eff._y_coefficient == int(4096 * ng.nextgen_damper)

        ng.spring_mode = SpringModeEnum.BASIC
        _, telem2 = self._make(SpringModeEnum.BASIC)
        ng.on_telemetry(telem2)
        assert not ng.effects["nextgen_damper"].started

    def test_pedals_damper_is_x_only(self):
        ng, telem = self._make(SpringModeEnum.NEXTGEN, ffb_type="pedals")
        ng.on_telemetry(telem)
        eff = ng.effects["nextgen_damper"]
        assert eff.started
        assert eff._x_coefficient == int(4096 * ng.nextgen_damper)
        assert eff._y_coefficient == 0

    def test_rudder_slip_force_not_speed_faded_in_nextgen(self):
        # BASIC scales the slip constant-force by IAS/Vne (the documented V^3
        # divergence); NEXTGEN skips the extra linear ramp.
        basic, telem_b = self._make(SpringModeEnum.BASIC, ffb_type="pedals")
        ng, telem_n = self._make(SpringModeEnum.NEXTGEN, ffb_type="pedals")
        for inst in (basic, ng):
            inst.rudder_force_dampener.update = lambda v, **kw: v
        args = dict(slip_angle=0.1, rudder_angle=0.0,
                    _dyn_pressure=1.0, _slip_gain=1.0, vne=100.0)
        f_basic = basic._calculate_rudder_force(telem_b, **args)
        f_ng = ng._calculate_rudder_force(telem_n, **args)
        # telem IAS is 50 m/s = half Vne: BASIC halves the force, NEXTGEN doesn't
        assert f_ng == pytest.approx(2.0 * f_basic, abs=1e-9)
        assert f_ng == pytest.approx(0.1 * ng.rudder_gain, abs=1e-9)
