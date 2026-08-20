"""Feature engineering (regime normalization + leak-safe rolling-window summaries)."""
from pdm.features.flight import (
    add_flight_deltas,
    flight_summary,
    make_flight_windows,
    make_within_flight_windows,
    reduce_to_flights,
)
from pdm.features.regime import RegimeNormalizer
from pdm.features.windows import (
    add_rolling_features,
    feature_columns,
    last_per_unit,
)

__all__ = [
    "RegimeNormalizer",
    "add_flight_deltas",
    "add_rolling_features",
    "feature_columns",
    "flight_summary",
    "last_per_unit",
    "make_flight_windows",
    "make_within_flight_windows",
    "reduce_to_flights",
]
