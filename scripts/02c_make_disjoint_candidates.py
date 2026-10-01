# scripts/02c_make_disjoint_candidates.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

INP = Path("data/url_candidates.csv")
OUT = Path("data/url_candidates_disjoint.csv")
OUT_OVERLAPS = Path("data/url_overlaps.csv")

def main():
    df = pd.read_csv(INP)

    # sanity
    assert {"portal", "topic", "url"}.issubset(df.columns)

    # how many topics does each (portal,url) belong to?
    g = (
        df.groupby(["portal", "url"])["topic"]
        .nunique()
        .reset_index(name="n_topics")
    )

    overlaps = g[g["n_topics"] > 1].copy()
    print(f"Input rows: {len(df)}")
    print(f"Unique (portal,url): {df.drop_duplicates(['portal','url']).shape[0]}")
    print(f"Overlapping (portal,url) with >1 topic: {len(overlaps)}")

    # Save overlaps for inspection
    if len(overlaps) > 0:
        # include the actual topic list for each overlap
        topic_list = (
            df.groupby(["portal", "url"])["topic"]
            .apply(lambda s: sorted(set(s)))
            .reset_index(name="topics")
        )
        ov = overlaps.merge(topic_list, on=["portal", "url"], how="left")
        ov.to_csv(OUT_OVERLAPS, index=False)
        print(f"Saved overlaps report: {OUT_OVERLAPS}")

    # Drop all rows whose (portal,url) is overlapping
    df2 = df.merge(overlaps[["portal", "url"]], on=["portal", "url"], how="left", indicator=True)
    df2 = df2[df2["_merge"] == "left_only"].drop(columns=["_merge"])

    # Optional: also dedup identical rows (should already be unique)
    df2 = df2.drop_duplicates()

    df2.to_csv(OUT, index=False)
    print(f"Saved disjoint candidates: {OUT} (rows={len(df2)})")
    if len(df2):
        print(df2.groupby(["portal", "topic"]).size())

if __name__ == "__main__":
    main()
