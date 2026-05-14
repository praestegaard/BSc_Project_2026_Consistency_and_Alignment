"""
Spearman correlation between consistency metrics and cross-alignment rate.
"""

import logging
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

FILE_PATH = "results/cross_algorithm_analysis.xlsx"

ALIGNED_COL = "Aligned (Cross)"
MISALIGNED_COL = "Misaligned (Cross)"

METRICS = ["Jaccard", "Cosine", "SequenceMatcher", "Levenshtein", "PubMedBERT"]


def load_all_sheets(file_path):
    sheets = pd.read_excel(file_path, sheet_name=None)
    frames = []
    for sheet_name, df in sheets.items():
        df = df.copy()
        df["Sheet"] = sheet_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Question", "Model"], how="all")
    return combined


def calculate_spearman(df, group_name="Overall"):
    results = []
    for metric in METRICS:
        subset = df[[metric, "Cross Alignment Rate"]].dropna()
        rho, p_value = spearmanr(subset[metric], subset["Cross Alignment Rate"])
        results.append({
            "Group": group_name,
            "Metric": metric,
            "n": len(subset),
            "Spearman rho": round(rho, 4),
            "p-value": round(p_value, 4),
        })
    return pd.DataFrame(results)


def run():
    df = load_all_sheets(FILE_PATH)
    logger.info("Loaded %d rows, columns: %s", len(df), df.columns.tolist())

    # cross alignment rate
    total = df[ALIGNED_COL] + df[MISALIGNED_COL]
    df["Cross Alignment Rate"] = df[ALIGNED_COL] / total

    overall = calculate_spearman(df, group_name="Overall")
    logger.info("Overall:\n%s", overall.to_string(index=False))

    by_type = pd.concat([
        calculate_spearman(group_df, group_name=qtype)
        for qtype, group_df in df.groupby("Type")
    ], ignore_index=True)
    logger.info("By question type:\n%s", by_type.to_string(index=False))

    by_model = pd.concat([
        calculate_spearman(group_df, group_name=sheet)
        for sheet, group_df in df.groupby("Sheet")
    ], ignore_index=True)
    logger.info("By model:\n%s", by_model.to_string(index=False))

    with pd.ExcelWriter("results/spearman_results.xlsx") as writer:
        overall.to_excel(writer, sheet_name="Overall", index=False)
        by_type.to_excel(writer, sheet_name="By Question Type", index=False)
        by_model.to_excel(writer, sheet_name="By Model", index=False)

    logger.info("Results saved → spearman_results.xlsx")


if __name__ == "__main__":
    run()