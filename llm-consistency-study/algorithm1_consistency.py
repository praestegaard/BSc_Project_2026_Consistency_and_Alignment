"""
Algorithm 1 — Consistency Analysis

Send each question K times to each model, then compute pairwise similarity across the five metrics.
Pairs that already have K responses registered are skipped, so interrupted runs can resume.
"""

import json
import logging
import os
from itertools import combinations

from config import (
    MODELS, K,
    QUESTIONS_FILE, ALG1_RESPONSES_FILE, ALG1_SCORES_FILE, RESULTS_DIR,
)
from llm_client import call_llm
from metrics.similarity import compute_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

METRIC_NAMES = ["jaccard", "cosine", "sequence_matcher", "levenshtein", "pubMedBert"]


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_existing_responses():
    if os.path.exists(ALG1_RESPONSES_FILE):
        with open(ALG1_RESPONSES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_responses(responses):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(ALG1_RESPONSES_FILE, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)


def collect_responses(questions):
    responses = load_existing_responses()
    total = len(questions) * len(MODELS) * K
    done = 0

    for q in questions:
        qid = q["id"]
        prompt = q["question"]

        for model_key in MODELS:
            key = f"q{qid}_{model_key}"
            existing = responses.get(key, {}).get("responses", [])

            if len(existing) >= K:
                done += K
                logger.info("Skipping %s — already have %d responses", key, len(existing))
                continue

            responses.setdefault(key, {
                "question_id": qid,
                "question": prompt,
                "question_type": q["type"],
                "model": model_key,
                "model_display": MODELS[model_key]["display_name"],
                "responses": existing,
            })

            for i in range(len(existing), K):
                done += 1
                logger.info("[%d/%d] %s  run %d/%d", done, total, key, i + 1, K)
                try:
                    text = call_llm(model_key, prompt)
                    responses[key]["responses"].append(text)
                except Exception as exc:
                    logger.error("FAILED %s run %d: %s", key, i + 1, exc)
                    responses[key]["responses"].append(None)

                # save after every call so nothing is lost on crash
                save_responses(responses)

    return responses


def compute_similarity_scores(responses):
    scores = {}

    for key, data in responses.items():
        resps = [r for r in data["responses"] if r is not None]
        if len(resps) < 2:
            logger.warning("< 2 valid responses for %s, skipping similarity", key)
            continue

        pair_scores = []
        for i, j in combinations(range(len(resps)), 2):
            logger.info("Similarity %s  pair (%d, %d)", key, i, j)
            pair_scores.append({"pair": [i, j], "scores": compute_all(resps[i], resps[j])})

        # average each metric across all C(K,2) pairs
        averages = {}
        for m in METRIC_NAMES:
            vals = [p["scores"][m] for p in pair_scores]
            averages[m] = round(sum(vals) / len(vals), 4) if vals else 0.0

        scores[key] = {
            "question_id": data["question_id"],
            "question_type": data["question_type"],
            "model": data["model"],
            "model_display": data["model_display"],
            "num_valid_responses": len(resps),
            "num_pairs": len(pair_scores),
            "pair_scores": pair_scores,
            "averages": averages,
        }

    return scores


def run():
    questions = load_questions()
    logger.info("Loaded %d questions", len(questions))

    logger.info("=== Collecting responses (K=%d) ===", K)
    responses = collect_responses(questions)

    logger.info("=== Computing pairwise similarity ===")
    scores = compute_similarity_scores(responses)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(ALG1_SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    logger.info("Done. Responses → %s   Scores → %s", ALG1_RESPONSES_FILE, ALG1_SCORES_FILE)


if __name__ == "__main__":
    run()