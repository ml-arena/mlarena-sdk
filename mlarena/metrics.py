"""How a metric value reads: the console's one formatter
(`frontend/src/utils/metricFormat.ts`), for a notebook or a script.

A value is shown through its descriptor (a `MetricSpec`, e.g. one of
`leaderboard()`'s `df.attrs["metrics"]`): `format`, `unit` and `precision` are
the only presentation inputs. The value itself stays a number everywhere the
API serves it; this module never changes a payload.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

# The platform's `metrics_schema.FORMATS`, as of mlarena-sdk 4.15.0.
_KNOWN_FORMATS = ("number", "integer", "percent", "seconds", "bytes", "currency", "energy", "mass")
_BYTE_UNITS = ("B", "KB", "MB", "GB", "TB")
# `energy` values are in Wh, `mass` values in g (the platform's
# `metrics_schema`): each is shown in the unit that puts it in [1, 1000).
_SCALED_UNITS = {
    "energy": (("mWh", 1e-3), ("Wh", 1.0), ("kWh", 1e3)),
    "mass": (("mg", 1e-3), ("g", 1.0), ("kg", 1e3)),
}
MISSING = "—"


def _fixed(value: float, precision: int) -> str:
    return f"{value:,.{precision}f}"


def _bytes(value: float, precision: int) -> str:
    if value == 0:
        return "0 B"
    i = min(len(_BYTE_UNITS) - 1, max(0, int(math.floor(math.log(abs(value), 1024)))))
    scaled = value / 1024 ** i
    # A whole size drops its decimals ("100 MB"), as the console shows it.
    return f"{_fixed(scaled, 0 if float(scaled).is_integer() else precision)} {_BYTE_UNITS[i]}"


def _scaled(value: float, precision: int, units: tuple) -> str:
    """mWh / Wh / kWh (mg / g / kg): the largest unit the value reaches, the
    base unit for 0; one unit up when rounding to `precision` reaches 1000
    (0.9999 Wh at 2 decimals is "1.00 Wh", not "1,000.00 mWh")."""
    base = next(i for i, (_, factor) in enumerate(units) if factor == 1.0)
    if value == 0:
        i = base
    else:
        i = 0
        for j, (_, factor) in enumerate(units):
            if abs(value) >= factor:
                i = j
        if i < len(units) - 1 and round(abs(value) / units[i][1], precision) >= 1000:
            i += 1
    name, factor = units[i]
    return f"{_fixed(value / factor, precision)} {name}"


def format_metric(value: Any, spec: Mapping[str, Any]) -> str:
    """`value` as the descriptor `spec` shows it.

    `number` / `integer` grouped by thousands, `percent` a fraction shown
    ×100, `seconds` with " s", `bytes` scaled B / KB / MB / GB / TB (base
    1024), `currency` with `spec["unit"]`, `energy` (a value in Wh) scaled
    mWh / Wh / kWh and `mass` (a value in g) scaled mg / g / kg. Every format
    uses `spec["precision"]` decimals (`integer` none; a whole `bytes` or
    `currency` value drops them). A non-currency `unit` belongs to the column
    header, not to the value. None or NaN (no value yet, or a run that was not
    metered) reads "—".

    Raises `ValueError` on a format this SDK does not know (a newer platform's:
    upgrade the SDK) and on a `currency` spec without a unit.
    """
    fmt = spec["format"]
    precision = spec["precision"]
    if fmt not in _KNOWN_FORMATS:
        raise ValueError(
            f"format_metric: unknown metric format {fmt!r} (known: "
            f"{list(_KNOWN_FORMATS)}); a newer platform's format needs a newer mlarena-sdk"
        )
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING
    value = float(value)
    if fmt == "number":
        return _fixed(value, precision)
    if fmt == "integer":
        return _fixed(round(value), 0)
    if fmt == "percent":
        return f"{_fixed(value * 100, precision)}%"
    if fmt == "seconds":
        return f"{_fixed(value, precision)} s"
    if fmt == "bytes":
        return _bytes(value, precision)
    if fmt == "currency":
        if spec["unit"] is None:
            raise ValueError("format_metric: a currency metric needs a unit")
        digits = _fixed(value, 0 if value.is_integer() else precision)
        return f"{digits} {spec['unit']}"
    return _scaled(value, precision, _SCALED_UNITS[fmt])
