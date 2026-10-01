"""
AF-XAI Step 2: dataset validation, preprocessing, and three simulated substations.

This script does not train One-Class SVM, aggregate models, or compute XAI/metrics.
It only prepares client datasets for later federated anomaly detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Paths and experiment constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "dataset" / "raw" / "power_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
SCALED_DIR = PROCESSED_DIR / "scaled"
MODELS_LOCAL_DIR = PROJECT_ROOT / "models" / "local"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

RANDOM_SEED = 42
N_CLIENTS = 3
TEST_SIZE = 0.20

REQUIRED_COLUMNS = [
    "node_id",
    "timestamp",
    "voltage",
    "frequency",
    "active_power",
    "reactive_power",
    "load_demand",
    "renewable_output",
    "temperature",
    "humidity",
    "energy_price",
    "power_loss",
    "region_id",
    "stability_index",
    "fault_indicator",
    "attack_label",
]

# Electrical / environmental measurements used by the later OCSVM.
# fault_indicator is ground truth for evaluation only and is never an input feature.
# attack_label is a cyber-attack tag and is out of scope for this electrical-fault study.
# node_id, timestamp, and region_id are metadata, not model inputs.
# energy_price is not part of the initial AF-XAI feature set.
ML_FEATURES = [
    "voltage",
    "frequency",
    "active_power",
    "reactive_power",
    "load_demand",
    "renewable_output",
    "temperature",
    "humidity",
    "power_loss",
    "stability_index",
]

METADATA_COLUMNS = ["node_id", "timestamp", "region_id", "fault_indicator"]
OUTPUT_COLUMNS = METADATA_COLUMNS + ML_FEATURES
EXCLUDED_COLUMNS = ["attack_label", "energy_price"]

EXPECTED = {
    "n_records": 20000,
    "n_nodes": 19,
    "n_regions": 4,
    "n_faults": 560,
    "n_normal": 19440,
}

CLIENT_IDS = [f"client_{i}" for i in range(1, N_CLIENTS + 1)]


def load_raw_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> dict:
    """Check schema, integrity, and published dataset facts. Does not alter values."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    n_missing = int(df.isna().sum().sum())
    n_duplicates = int(df.duplicated().sum())
    n_nodes = int(df["node_id"].nunique())
    n_regions = int(df["region_id"].nunique())
    n_faults = int((df["fault_indicator"] == 1).sum())
    n_normal = int((df["fault_indicator"] == 0).sum())

    if n_missing:
        raise ValueError(f"Dataset contains {n_missing} missing values.")
    if n_duplicates:
        raise ValueError(f"Dataset contains {n_duplicates} duplicate rows.")

    checks = {
        "required_columns_present": True,
        "n_records": int(len(df)),
        "n_missing_values": n_missing,
        "n_duplicate_rows": n_duplicates,
        "n_unique_nodes": n_nodes,
        "n_unique_regions": n_regions,
        "n_normal": n_normal,
        "n_faults": n_faults,
        "extra_columns": extra_cols,
        "matches_expected_facts": {
            "n_records": len(df) == EXPECTED["n_records"],
            "n_nodes": n_nodes == EXPECTED["n_nodes"],
            "n_regions": n_regions == EXPECTED["n_regions"],
            "n_faults": n_faults == EXPECTED["n_faults"],
            "n_normal": n_normal == EXPECTED["n_normal"],
        },
    }

    failed = [k for k, ok in checks["matches_expected_facts"].items() if not ok]
    if failed:
        raise ValueError(f"Dataset facts did not match expectations: {failed}. Details: {checks}")

    print("Dataset validation passed.")
    print(f"  records={len(df):,}  nodes={n_nodes}  regions={n_regions}")
    print(f"  normal={n_normal:,}  fault={n_faults:,}  missing={n_missing}  duplicates={n_duplicates}")
    return checks


def parse_and_sort_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps for chronological order. Timestamp is metadata, not an ML feature."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out = out.sort_values(["timestamp", "node_id", "region_id"], kind="mergesort").reset_index(drop=True)
    return out


def _partition_score(assign: dict[int, str], node_records: dict, node_faults: dict) -> tuple:
    """Lower is better: node-count range, then record range, then fault range."""
    recs, faults, counts = [], [], []
    for cid in CLIENT_IDS:
        nodes = [n for n, c in assign.items() if c == cid]
        counts.append(len(nodes))
        recs.append(sum(node_records[n] for n in nodes))
        faults.append(sum(node_faults[n] for n in nodes))
    return (max(counts) - min(counts), max(recs) - min(recs), max(faults) - min(faults))


def assign_nodes_to_clients(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Node-level partitioning: every row of a node_id stays on one client.

    Nineteen nodes cannot be split 6/6/6, so sizes are 7/6/6. Assignment is not
    region-equals-client. A greedy load-balance is refined by deterministic swaps;
    seed 42 is used only for tie-breaking when two clients look equal.
    """
    node_stats = (
        df.groupby("node_id", as_index=False)
        .agg(record_count=("node_id", "size"), fault_count=("fault_indicator", "sum"))
        .sort_values("node_id")
    )
    node_records = dict(zip(node_stats["node_id"], node_stats["record_count"]))
    node_faults = dict(zip(node_stats["node_id"], node_stats["fault_count"]))
    node_ids = list(node_stats["node_id"])

    rng = np.random.RandomState(seed)
    max_nodes = int(np.ceil(len(node_ids) / N_CLIENTS))  # 7
    min_nodes = int(np.floor(len(node_ids) / N_CLIENTS))  # 6

    # Largest-first greedy placement onto the currently lightest eligible client.
    ordered = node_stats.sort_values(
        ["record_count", "fault_count", "node_id"],
        ascending=[False, False, True],
    )
    assign: dict[int, str] = {}
    load_rec = {cid: 0 for cid in CLIENT_IDS}
    load_fault = {cid: 0 for cid in CLIENT_IDS}
    load_n = {cid: 0 for cid in CLIENT_IDS}

    remaining = len(node_ids)
    for _, row in ordered.iterrows():
        nid = int(row["node_id"])
        remaining -= 1
        eligible = []
        for cid in CLIENT_IDS:
            if load_n[cid] >= max_nodes:
                continue
            # Leave enough free slots so every client can still reach min_nodes.
            tentative = dict(load_n)
            tentative[cid] += 1
            slots_left_after = sum(max_nodes - tentative[c] for c in CLIENT_IDS)
            deficit = sum(max(0, min_nodes - tentative[c]) for c in CLIENT_IDS)
            if remaining >= deficit and slots_left_after >= remaining:
                eligible.append(cid)
        if not eligible:
            eligible = [cid for cid in CLIENT_IDS if load_n[cid] < max_nodes]

        def client_key(cid: str) -> tuple:
            # Prefer fewer records, then fewer faults, then fewer nodes.
            # Tie-break with a seeded random value so the choice is reproducible.
            return (
                load_rec[cid],
                load_fault[cid],
                load_n[cid],
                float(rng.random()),
                cid,
            )

        chosen = min(eligible, key=client_key)
        assign[nid] = chosen
        load_n[chosen] += 1
        load_rec[chosen] += int(row["record_count"])
        load_fault[chosen] += int(row["fault_count"])

    # Pairwise swaps if they improve balance without violating 6–7 nodes per client.
    improved = True
    while improved:
        improved = False
        current = _partition_score(assign, node_records, node_faults)
        for a in node_ids:
            for b in node_ids:
                if a >= b:
                    continue
                if assign[a] == assign[b]:
                    continue
                assign[a], assign[b] = assign[b], assign[a]
                cand = _partition_score(assign, node_records, node_faults)
                counts_ok = True
                for cid in CLIENT_IDS:
                    n_here = sum(1 for c in assign.values() if c == cid)
                    if n_here < min_nodes or n_here > max_nodes:
                        counts_ok = False
                        break
                if counts_ok and cand < current:
                    current = cand
                    improved = True
                else:
                    assign[a], assign[b] = assign[b], assign[a]

    # Mapping rows are (client, node, region) so region_id is preserved truthfully.
    # The client is still chosen at node_id level, not by region.
    mapping_rows = []
    for (nid, rid), grp in df.groupby(["node_id", "region_id"], sort=True):
        mapping_rows.append(
            {
                "client_id": assign[int(nid)],
                "node_id": int(nid),
                "region_id": int(rid),
                "record_count": int(len(grp)),
                "fault_count": int((grp["fault_indicator"] == 1).sum()),
            }
        )
    mapping = pd.DataFrame(mapping_rows).sort_values(
        ["client_id", "node_id", "region_id"]
    ).reset_index(drop=True)
    return mapping


def split_client_train_test(client_df: pd.DataFrame, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-client hold-out split for anomaly detection.

    One-Class SVM later learns the distribution of *normal* operation only
    (fault_indicator == 0). Fault rows are never training inputs; they are
    reserved for evaluation. Normal rows are split 80/20 with random_state=42.
    All fault rows from this client go to test.csv so they remain available
    for evaluation and cannot leak into scaler fitting or training.
    """
    normals = client_df.loc[client_df["fault_indicator"] == 0].copy()
    faults = client_df.loc[client_df["fault_indicator"] == 1].copy()

    if len(normals) < 2:
        raise ValueError("Client does not have enough normal samples to split.")

    train_normal, test_normal = train_test_split(
        normals,
        test_size=TEST_SIZE,
        random_state=seed,
        shuffle=True,
    )
    test_df = pd.concat([test_normal, faults], axis=0)
    train_normal = train_normal.sort_values(["timestamp", "node_id"]).reset_index(drop=True)
    test_df = test_df.sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    if int((train_normal["fault_indicator"] != 0).sum()) != 0:
        raise RuntimeError("Training set leaked fault samples.")
    return train_normal, test_df


def fit_and_apply_scaler(
    train_normal: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on this client's normal training features only."""
    scaler = StandardScaler()
    scaler.fit(train_normal[ML_FEATURES])

    train_scaled = train_normal[OUTPUT_COLUMNS].copy()
    test_scaled = test_df[OUTPUT_COLUMNS].copy()
    train_scaled[ML_FEATURES] = scaler.transform(train_normal[ML_FEATURES])
    test_scaled[ML_FEATURES] = scaler.transform(test_df[ML_FEATURES])
    return scaler, train_scaled, test_scaled


def format_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def save_client_outputs(
    client_id: str,
    train_normal: pd.DataFrame,
    test_df: pd.DataFrame,
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    scaler: StandardScaler,
    mapping: pd.DataFrame,
) -> dict:
    client_dir = PROCESSED_DIR / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    SCALED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    train_path = client_dir / "train_normal.csv"
    test_path = client_dir / "test.csv"
    meta_path = client_dir / "metadata.json"
    scaler_path = MODELS_LOCAL_DIR / f"{client_id}_scaler.joblib"
    train_scaled_path = SCALED_DIR / f"{client_id}_train_scaled.csv"
    test_scaled_path = SCALED_DIR / f"{client_id}_test_scaled.csv"

    format_timestamp_column(train_normal[OUTPUT_COLUMNS]).to_csv(train_path, index=False)
    format_timestamp_column(test_df[OUTPUT_COLUMNS]).to_csv(test_path, index=False)
    format_timestamp_column(train_scaled[OUTPUT_COLUMNS]).to_csv(train_scaled_path, index=False)
    format_timestamp_column(test_scaled[OUTPUT_COLUMNS]).to_csv(test_scaled_path, index=False)
    joblib.dump(scaler, scaler_path)

    client_map = mapping.loc[mapping["client_id"] == client_id]
    node_ids = sorted(client_map["node_id"].unique().tolist())
    region_ids = sorted(client_map["region_id"].unique().tolist())
    n_records = int(client_map["record_count"].sum())
    n_faults = int(client_map["fault_count"].sum())
    n_normal = n_records - n_faults

    metadata = {
        "client_id": client_id,
        "node_ids": node_ids,
        "region_ids": region_ids,
        "n_nodes": len(node_ids),
        "n_records": n_records,
        "n_normal": n_normal,
        "n_fault": n_faults,
        "train_normal_records": int(len(train_normal)),
        "test_records": int(len(test_df)),
        "test_normal_records": int((test_df["fault_indicator"] == 0).sum()),
        "test_fault_records": int((test_df["fault_indicator"] == 1).sum()),
        "ml_features": ML_FEATURES,
        "metadata_columns": METADATA_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "notes": {
            "fault_indicator": (
                "Ground-truth evaluation label only. Never used as an ML input feature."
            ),
            "attack_label": (
                "Excluded. Current AF-XAI task is electrical/grid fault detection, "
                "not cyberattack classification."
            ),
            "ocsvm_training": (
                "train_normal.csv contains only fault_indicator == 0 rows so the later "
                "One-Class SVM learns normal operating behaviour."
            ),
            "split": (
                "Normal samples are split 80/20 (random_state=42). All fault samples "
                "from this client are placed in test.csv for evaluation."
            ),
            "scaling": (
                "StandardScaler fitted only on this client's normal training features. "
                "Original unscaled measurements are stored in train_normal.csv and test.csv."
            ),
        },
        "scaler_path": str(scaler_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "scaling_method": "StandardScaler",
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "files": {
            "train_normal": str(train_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "test": str(test_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "train_scaled": str(train_scaled_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "test_scaled": str(test_scaled_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def write_reports(validation: dict, mapping: pd.DataFrame, client_metas: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path = REPORTS_DIR / "client_node_mapping.csv"
    mapping.to_csv(mapping_path, index=False)

    client_statistics = []
    for meta in client_metas:
        client_statistics.append(
            {
                "client_id": meta["client_id"],
                "n_nodes": meta["n_nodes"],
                "node_ids": meta["node_ids"],
                "n_records": meta["n_records"],
                "n_normal": meta["n_normal"],
                "n_fault": meta["n_fault"],
                "train_normal_records": meta["train_normal_records"],
                "test_records": meta["test_records"],
                "test_normal_records": meta["test_normal_records"],
                "test_fault_records": meta["test_fault_records"],
            }
        )

    total_records = validation["n_records"]
    total_faults = validation["n_faults"]
    report = {
        "total_records": total_records,
        "total_nodes": validation["n_unique_nodes"],
        "total_regions": validation["n_unique_regions"],
        "total_normal_samples": validation["n_normal"],
        "total_fault_samples": total_faults,
        "fault_percentage": round(100.0 * total_faults / total_records, 4),
        "selected_features": ML_FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS + ["node_id", "timestamp", "region_id", "fault_indicator"],
        "excluded_from_ml_inputs_reason": {
            "fault_indicator": "evaluation label, not a feature",
            "attack_label": "cyberattack label, out of scope",
            "node_id": "metadata",
            "timestamp": "metadata (used only for sorting)",
            "region_id": "metadata",
            "energy_price": "not in the initial AF-XAI feature set",
        },
        "client_statistics": client_statistics,
        "training_test_counts": {
            meta["client_id"]: {
                "train_normal": meta["train_normal_records"],
                "test": meta["test_records"],
            }
            for meta in client_metas
        },
        "normal_training_counts": {
            meta["client_id"]: meta["train_normal_records"] for meta in client_metas
        },
        "test_normal_fault_counts": {
            meta["client_id"]: {
                "test_normal": meta["test_normal_records"],
                "test_fault": meta["test_fault_records"],
            }
            for meta in client_metas
        },
        "scaling_method": "StandardScaler (fit on each client's normal training samples only)",
        "random_seed": RANDOM_SEED,
        "raw_dataset": str(RAW_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "raw_dataset_unchanged": True,
    }
    (REPORTS_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_rows = []
    for meta in client_metas:
        summary_rows.append(
            {
                "client_id": meta["client_id"],
                "n_nodes": meta["n_nodes"],
                "node_ids": " ".join(str(n) for n in meta["node_ids"]),
                "record_count": meta["n_records"],
                "normal_count": meta["n_normal"],
                "fault_count": meta["n_fault"],
                "train_normal_count": meta["train_normal_records"],
                "test_count": meta["test_records"],
                "test_normal_count": meta["test_normal_records"],
                "test_fault_count": meta["test_fault_records"],
                "scaling_method": "StandardScaler",
                "random_seed": RANDOM_SEED,
            }
        )
    pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "dataset_summary.csv", index=False)


def print_client_statistics(mapping: pd.DataFrame, client_metas: list[dict]) -> None:
    print("\nClient-to-node mapping (node_id is assigned wholly to one client):")
    node_level = (
        mapping.groupby(["client_id", "node_id"], as_index=False)
        .agg(record_count=("record_count", "sum"), fault_count=("fault_count", "sum"))
        .sort_values(["client_id", "node_id"])
    )
    print(node_level.to_string(index=False))

    print("\nRecords / faults per client:")
    totals = mapping.groupby("client_id", as_index=False).agg(
        n_nodes=("node_id", "nunique"),
        record_count=("record_count", "sum"),
        fault_count=("fault_count", "sum"),
    )
    print(totals.to_string(index=False))

    print("\nTrain / test counts per client:")
    for meta in client_metas:
        train_faults = 0
        print(
            f"  {meta['client_id']}: train_normal={meta['train_normal_records']} "
            f"(faults_in_train={train_faults})  "
            f"test={meta['test_records']} "
            f"(normal={meta['test_normal_records']}, fault={meta['test_fault_records']})"
        )
        print("    Training contains only normal samples: YES")


def main() -> None:
    print("AF-XAI preprocessing (Step 2)")
    print(f"Loading {RAW_PATH.relative_to(PROJECT_ROOT)}")
    raw = load_raw_dataset(RAW_PATH)
    validation = validate_dataset(raw)

    df = parse_and_sort_timestamps(raw)
    mapping = assign_nodes_to_clients(df, seed=RANDOM_SEED)

    node_to_client = (
        mapping[["node_id", "client_id"]]
        .drop_duplicates()
        .set_index("node_id")["client_id"]
        .to_dict()
    )
    df = df.copy()
    df["client_id"] = df["node_id"].map(node_to_client)

    client_metas = []
    for client_id in CLIENT_IDS:
        client_df = df.loc[df["client_id"] == client_id].copy()
        train_normal, test_df = split_client_train_test(client_df, seed=RANDOM_SEED)
        scaler, train_scaled, test_scaled = fit_and_apply_scaler(train_normal, test_df)
        meta = save_client_outputs(
            client_id=client_id,
            train_normal=train_normal,
            test_df=test_df,
            train_scaled=train_scaled,
            test_scaled=test_scaled,
            scaler=scaler,
            mapping=mapping,
        )
        client_metas.append(meta)

    write_reports(validation, mapping, client_metas)
    print_client_statistics(mapping, client_metas)

    print("\nGenerated files:")
    generated = [
        REPORTS_DIR / "client_node_mapping.csv",
        REPORTS_DIR / "dataset_report.json",
        REPORTS_DIR / "dataset_summary.csv",
    ]
    for client_id in CLIENT_IDS:
        generated.extend(
            [
                PROCESSED_DIR / client_id / "train_normal.csv",
                PROCESSED_DIR / client_id / "test.csv",
                PROCESSED_DIR / client_id / "metadata.json",
                MODELS_LOCAL_DIR / f"{client_id}_scaler.joblib",
                SCALED_DIR / f"{client_id}_train_scaled.csv",
                SCALED_DIR / f"{client_id}_test_scaled.csv",
            ]
        )
    for path in generated:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        print(f"  {rel}")

    print("\nRaw CSV left unchanged. No OCSVM, aggregation, SHAP, LIME, Flask, or React was run.")


if __name__ == "__main__":
    main()
