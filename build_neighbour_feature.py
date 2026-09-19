"""
Step 6: Build the upwind-neighbor "knowledge" feature.

For each (turbine, timestamp), find the nearest OTHER turbine at the same
site that currently lies upwind (within TOLERANCE_DEG of this turbine's
measured wind direction), and pull that neighbor's simultaneous Power and
Wind speed as the candidate "knowledge" signal for CAKI's graph/relational
path. This is what r1 (knowledge-injected route) will use that r0 (base
route) does not have.

Geometry: turbine lat/lon -> local flat-earth (x=east, y=north) meters,
via equirectangular projection around the site centroid (valid at this
scale, sites are ~1-2 km across). Bearing from turbine A to turbine B is
computed in standard compass convention (0=N, 90=E, clockwise). A turbine
B is "upwind" of A if the bearing from A to B roughly matches A's measured
wind direction (wind direction = the direction the wind is blowing FROM).

Reports how often a geometrically valid AND data-available upwind neighbor
exists -- this is the ceiling on how often the knowledge route can even be
applied, which matters directly for the CAKI experiment (knowledge must
NOT always be available/useful, or there's nothing for the gate to decide).

"""

from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
LABELED_PATH = ROOT_DIR / "Results" / "labeled_dataset.csv"
OUT_PATH = ROOT_DIR / "Results" / "labeled_with_neighbor.csv"

TOLERANCE_DEG = 30.0   # half-width of the upwind acceptance cone


def load_static_metadata():
    frames = []
    for site_dir, fname in [("Kelmarsh", "Kelmarsh_WT_static.csv"),
                             ("Penmanshiel", "Penmanshiel_WT_static.csv")]:
        path = DATA_DIR / site_dir / fname
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.dropna(subset=["Title"])
        df["turbine_id"] = df["Title"].str.extract(r"(\d+)$").astype(int).astype(str)
        df["site"] = site_dir
        frames.append(df[["site", "turbine_id", "Latitude", "Longitude"]])
    return pd.concat(frames, ignore_index=True)


def compute_site_geometry(meta_site: pd.DataFrame):
    """Returns (turbine_ids list, bearing_matrix, dist_matrix) for one site."""
    lat0 = meta_site["Latitude"].mean()
    lon0 = meta_site["Longitude"].mean()
    R = 6371000.0

    x = R * np.radians(meta_site["Longitude"].values - lon0) * np.cos(np.radians(lat0))
    y = R * np.radians(meta_site["Latitude"].values - lat0)
    turbine_ids = meta_site["turbine_id"].values
    n = len(turbine_ids)

    dx = x[None, :] - x[:, None]   # dx[i,j] = x_j - x_i
    dy = y[None, :] - y[:, None]
    bearing = np.degrees(np.arctan2(dx, dy)) % 360.0   # bearing[i,j]: from i to j
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)  # exclude self as a candidate

    return turbine_ids, bearing, dist


def find_upwind_neighbor_for_site(df_site: pd.DataFrame, turbine_ids, bearing, dist):
    id_to_idx = {tid: i for i, tid in enumerate(turbine_ids)}
    idx_array = df_site["turbine_id"].map(id_to_idx).values
    wind_dir = df_site["Wind direction (\u00b0)"].values

    candidate_bearing = bearing[idx_array, :]          # (n_rows, n_turbines)
    candidate_dist = dist[idx_array, :]

    angular_diff = ((wind_dir[:, None] - candidate_bearing + 180) % 360) - 180
    valid = np.abs(angular_diff) <= TOLERANCE_DEG
    valid &= np.isfinite(candidate_dist)  # redundant with diagonal=inf, kept for clarity

    masked_dist = np.where(valid, candidate_dist, np.inf)
    best_idx = np.argmin(masked_dist, axis=1)
    has_geometric_neighbor = np.isfinite(masked_dist[np.arange(len(df_site)), best_idx])

    neighbor_tid = np.array(turbine_ids)[best_idx]
    neighbor_tid = np.where(has_geometric_neighbor, neighbor_tid, None)
    neighbor_dist = np.where(has_geometric_neighbor,
                              candidate_dist[np.arange(len(df_site)), best_idx], np.nan)

    out = df_site.copy()
    out["neighbor_turbine_id"] = neighbor_tid
    out["neighbor_distance_m"] = neighbor_dist
    out["has_geometric_neighbor"] = has_geometric_neighbor
    return out


def main():
    print("Loading turbine metadata and labeled dataset...")
    meta = load_static_metadata()
    df = pd.read_csv(LABELED_PATH, parse_dates=["timestamp"])
    df["turbine_id"] = df["turbine_id"].astype(str)
    print(f"Loaded {len(df)} rows.")

    results = []
    for site, df_site in df.groupby("site"):
        # Restrict candidate neighbors to turbines we actually have SCADA
        # data for at this site -- Penmanshiel's static file lists 14
        # turbines, but we only downloaded telemetry for 5 (11-15). Using
        # the full static list would let the algorithm pick "nearest
        # upwind neighbor" from turbines we have zero data for, which
        # would then always show as unavailable for reasons that have
        # nothing to do with wind geometry.
        present_turbines = df_site["turbine_id"].unique()
        meta_site = meta[(meta["site"] == site) & (meta["turbine_id"].isin(present_turbines))]
        print(f"\n{site}: {len(present_turbines)} turbines with data "
              f"(static file lists {(meta['site'] == site).sum()} total)")

        turbine_ids, bearing, dist = compute_site_geometry(meta_site)
        print(f"  -> {int((~np.isinf(dist)).sum())} directed turbine pairs among turbines with data")
        out = find_upwind_neighbor_for_site(df_site.reset_index(drop=True),
                                             turbine_ids, bearing, dist)
        results.append(out)

    df = pd.concat(results, ignore_index=True)

    pct_geo = 100 * df["has_geometric_neighbor"].mean()
    print(f"\nRows with a GEOMETRICALLY valid upwind neighbor (within "
          f"{TOLERANCE_DEG} deg): {df['has_geometric_neighbor'].sum()} ({pct_geo:.1f}%)")

    # Now join the neighbor's OWN simultaneous readings. Build a lookup keyed
    # by (site, turbine_id, timestamp) and join it in as the neighbor's data.
    lookup = df[["site", "turbine_id", "timestamp", "Power (kW)", "Wind speed (m/s)"]].copy()
    lookup = lookup.rename(columns={
        "turbine_id": "neighbor_turbine_id",
        "Power (kW)": "neighbor_power_kw",
        "Wind speed (m/s)": "neighbor_windspeed",
    })

    df = df.merge(lookup, on=["site", "neighbor_turbine_id", "timestamp"], how="left")

    df["neighbor_available"] = df["has_geometric_neighbor"] & df["neighbor_power_kw"].notna()
    pct_avail = 100 * df["neighbor_available"].mean()
    print(f"Rows with a geometrically valid AND data-available upwind neighbor: "
          f"{df['neighbor_available'].sum()} ({pct_avail:.1f}%)")

    print("\n--- Neighbor availability by site ---")
    print(df.groupby("site")["neighbor_available"].mean().mul(100).round(1).to_string())

    print("\n--- Neighbor availability by turbine ---")
    print(df.groupby(["site", "turbine_id"])["neighbor_available"].mean().mul(100).round(1).to_string())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    main()