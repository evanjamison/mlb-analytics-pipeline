#!/usr/bin/env python
# scripts/merge_base_into_combined.py
import sys
import pandas as pd


def main(base_path: str, combined_path: str) -> None:
    # Read base
    dfb = pd.read_csv(base_path, parse_dates=["game_date"], low_memory=False)

    # Read existing combined (or empty if missing)
    try:
        dfc = pd.read_csv(combined_path, parse_dates=["game_date"], low_memory=False)
    except FileNotFoundError:
        dfc = pd.DataFrame(columns=dfb.columns)

    # Align columns
    cols = sorted(set(dfb.columns).union(dfc.columns))
    dfb = dfb.reindex(columns=cols)
    dfc = dfc.reindex(columns=cols)

    # Merge + dedup by game_pk if present
    out = pd.concat([dfb, dfc], ignore_index=True)
    if "game_pk" in out.columns:
        out = (
            out.sort_values(["game_date", "game_pk"])
               .drop_duplicates("game_pk", keep="last")
        )
    else:
        out = out.drop_duplicates()

    out = out.sort_values("game_date")
    out.to_csv(combined_path, index=False)
    print(f"[merge] combined rows: {len(out):,}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: merge_base_into_combined.py BASE_CSV COMBINED_CSV"
        )
    main(sys.argv[1], sys.argv[2])
