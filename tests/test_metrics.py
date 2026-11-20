"""`mlarena.format_metric`: a value as the console's `formatMetric` shows it
under its MetricSpec, with the footprint formats `energy` (Wh) and `mass` (g)
auto-scaled (4.15.0)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena import format_metric


def spec(fmt, precision=2, unit=None):
    return {"key": "k", "label": "K", "source": "platform", "agg": "mean", "order": "asc",
            "format": fmt, "unit": unit, "precision": precision, "is_ranking": False,
            "visible": True}


@pytest.mark.parametrize("value, precision, expected", [
    (0.0042, 2, "4.20 mWh"),
    (0.00001, 3, "0.010 mWh"),
    (0.5, 1, "500.0 mWh"),
    (1.0, 2, "1.00 Wh"),
    (7.25, 2, "7.25 Wh"),
    (999.4, 1, "999.4 Wh"),
    (1000.0, 2, "1.00 kWh"),
    (2_500_000.0, 0, "2,500 kWh"),
    (0.0, 2, "0.00 Wh"),
    # Rounding that reaches 1000 moves one unit up, never "1,000.00 mWh".
    (0.999996, 2, "1.00 Wh"),
    (999.996, 2, "1.00 kWh"),
])
def test_energy_scales_mwh_wh_kwh(value, precision, expected):
    assert format_metric(value, spec("energy", precision)) == expected


@pytest.mark.parametrize("value, precision, expected", [
    (0.0123, 1, "12.3 mg"),
    (3.0, 2, "3.00 g"),
    (1530.0, 1, "1.5 kg"),
    (0, 0, "0 g"),
])
def test_mass_scales_mg_g_kg(value, precision, expected):
    assert format_metric(value, spec("mass", precision)) == expected


def test_unmetered_reads_a_dash():
    """A None (a run that was not metered, a submission with no metered run)
    is never shown as zero."""
    for fmt in ("energy", "mass", "seconds", "number"):
        assert format_metric(None, spec(fmt)) == "—"
    assert format_metric(math.nan, spec("energy")) == "—"


@pytest.mark.parametrize("fmt, value, precision, unit, expected", [
    ("number", 12345.678, 2, "mm", "12,345.68"),
    ("integer", 41.6, 4, None, "42"),
    ("percent", 0.1234, 1, None, "12.3%"),
    ("seconds", 0.12345, 3, None, "0.123 s"),
    ("bytes", 104857600, 2, None, "100 MB"),
    ("bytes", 1536, 1, None, "1.5 KB"),
    ("bytes", 0, 2, None, "0 B"),
    ("currency", 1234.5, 2, "€", "1,234.50 €"),
    ("currency", 40, 2, "€", "40 €"),
])
def test_the_console_formats(fmt, value, precision, unit, expected):
    assert format_metric(value, spec(fmt, precision, unit)) == expected


def test_unknown_format_raises_naming_it():
    with pytest.raises(ValueError, match="'joules'"):
        format_metric(1.0, spec("joules"))


def test_currency_without_unit_raises():
    with pytest.raises(ValueError, match="unit"):
        format_metric(1.0, spec("currency"))


def test_exported_from_the_package():
    assert mlarena.format_metric is format_metric
