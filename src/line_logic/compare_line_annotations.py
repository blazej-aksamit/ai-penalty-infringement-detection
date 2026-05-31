from pathlib import Path
import argparse
import pandas as pd


def normalize_label(x: str):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    mapping = {
        "on": "on_line",
        "online": "on_line",
        "on_line": "on_line",
        "off": "off_line",
        "offline": "off_line",
        "off_line": "off_line",
        "uncertain": "uncertain",
    }
    return mapping.get(x, x)


def find_friend_label_column(df: pd.DataFrame):
    candidates = [
        "expected_decision",
        "decision",
        "label",
        "friend_label",
        "manual_label",
        "result",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-csv", required=True)
    parser.add_argument("--friend-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    system_csv = Path(args.system_csv)
    friend_csv = Path(args.friend_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sys_df = pd.read_csv(system_csv)
    fr_df = pd.read_csv(friend_csv)

    if "image_name" not in sys_df.columns:
        raise ValueError("System CSV must contain 'image_name' column.")

    if "image_name" not in fr_df.columns:
        raise ValueError("Friend CSV must contain 'image_name' column.")

    friend_label_col = find_friend_label_column(fr_df)
    if friend_label_col is None:
        raise ValueError(
            f"Could not find friend label column. Friend CSV columns: {list(fr_df.columns)}"
        )

    sys_df = sys_df.copy()
    fr_df = fr_df.copy()

    sys_df["system_label"] = sys_df["decision"].map(normalize_label)
    fr_df["friend_label"] = fr_df[friend_label_col].map(normalize_label)

    keep_friend_cols = ["image_name", "friend_label"]
    extra_friend_cols = [c for c in fr_df.columns if c not in keep_friend_cols]
    merged = fr_df[keep_friend_cols + extra_friend_cols].merge(
        sys_df,
        on="image_name",
        how="left",
    )

    def classify(row):
        friend = row.get("friend_label", "")
        system = row.get("system_label", "")

        if system == "":
            return "missing_system_output"
        if friend == system:
            return "match"
        if system == "uncertain":
            return "conservative_uncertain"
        return "mismatch"

    merged["status"] = merged.apply(classify, axis=1)

    cols_front = [
        "image_name",
        "friend_label",
        "system_label",
        "status",
        "reason",
        "min_dist_px",
    ]
    remaining = [c for c in merged.columns if c not in cols_front]
    merged = merged[cols_front + remaining]

    merged.to_csv(out_csv, index=False)

    print(f"Saved comparison CSV: {out_csv}")
    print("\nStatus counts:")
    print(merged["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()