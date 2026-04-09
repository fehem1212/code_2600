import numpy as np


def add_features(df):
    out = df.copy()
    out["rooms_per_household"] = out["total_rooms"] / out["households"]
    out["population_per_household"] = out["population"] / out["households"]
    out["log_rooms"] = np.log1p(out["total_rooms"])
    out["log_population"] = np.log1p(out["population"])
    out["log_households"] = np.log1p(out["households"])
    out["lat_lon"] = out["latitude"] * out["longitude"]
    out["age_income"] = out["housing_median_age"] * out["median_income"]
    out["income_x_lon"] = out["median_income"] * out["longitude"]

    for col in out.columns:
        if getattr(out[col], "dtype", None) is not None and out[col].dtype.kind in "fc":
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = out[col].fillna(out[col].median())

    return out
