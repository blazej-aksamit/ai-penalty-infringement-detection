from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    docs = [
        root / "README.md",
        root / "docs" / "PROJECT_MAP.md",
        root / "docs" / "CANONICAL_WORKFLOW.md",
        root / "data" / "README.md",
        root / "scripts" / "README.md",
        root / "runs" / "README.md",
    ]

    print("Penalty Keeper Detection")
    print()
    print("Project navigation helper")
    print()
    print("Read these files first:")
    for path in docs:
        print(f"- {path}")
    print()
    print("Canonical metadata:")
    print(f"- {root / 'data' / 'meta' / 'kick_times.csv'}")
    print(f"- {root / 'data' / 'meta' / 'kick_windows_720p.csv'}")
    print(f"- {root / 'data' / 'meta' / 'keeper_violation_labels_final.csv'}")
    print(f"- {root / 'data' / 'meta' / 'splits_violation.csv'}")
    print()
    print("Canonical implementation areas:")
    print(f"- {root / 'scripts' / 'pipeline'}")
    print(f"- {root / 'scripts' / 'kick_detection'}")
    print(f"- {root / 'scripts' / 'line_logic'}")
    print(f"- {root / 'scripts' / 'ml'}")
    print()
    print("Legacy areas kept for reference:")
    print(f"- {root / 'scripts' / 'archive'}")
    print(f"- {root / 'scripts' / 'archive' / 'pipeline_nested_legacy'}")
    print(f"- {root / 'scripts' / 'archive' / 'line_logic_nested_legacy'}")


if __name__ == "__main__":
    main()
