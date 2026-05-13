"""
Shared evaluation loop for Algorithms 2 and 3.

Builds the prompt, calls the evaluator, parses the result, saves after
every single call so we can resume if anything crashes mid-run.
"""

import json
import os
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import MODELS, QUESTIONS_FILE, ALG1_RESPONSES_FILE, RESULTS_DIR
from llm_client import call_llm
from evaluation import (
    EVALUATION_SYSTEM_PROMPT,
    build_evaluation_prompt,
    parse_evaluation_response,
    compute_summary,
)

logger = logging.getLogger(__name__)


def count_must_have_points(must_have_str):
    """Count numbered lines, from beginning to end, in the must-have block."""
    if not must_have_str:
        return 0
    return len([
        line for line in must_have_str.split("\n")
        if line.strip() and re.match(r'^\d+\.', line.strip())
    ])


def _model_file(prefix, model_key):
    return os.path.join(RESULTS_DIR, f"{prefix}_{model_key}.json")


def load_results(prefix, model_key):
    path = _model_file(prefix, model_key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(prefix, model_key, results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(_model_file(prefix, model_key), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def evaluate_responses(result_key, results, evaluator_model, responses,
                       question_text, physician_answer, must_have,
                       total_points, entry_metadata, save_fn, log_prefix):
    """
    Run evaluator_model on each response, append results, and save
    after every call. Returns (completed, total).
    """
    if result_key in results and len(results[result_key].get("evaluations", [])) >= len(responses):
        logger.info("[%s] Skipping %s — already done", log_prefix, result_key)
        return 0, 0

    results.setdefault(result_key, {**entry_metadata, "evaluations": []})
    existing_count = len(results[result_key]["evaluations"])
    completed = 0
    total = 0

    for i in range(existing_count, len(responses)):
        total += 1
        logger.info("[%s] Evaluating %s  response %d/%d",
                    log_prefix, result_key, i + 1, len(responses))

        prompt = build_evaluation_prompt(
            question_text, physician_answer, responses[i],
            must_have_statements=must_have,
        )
        try:
            raw = call_llm(evaluator_model, prompt, system_prompt=EVALUATION_SYSTEM_PROMPT)
            parsed = parse_evaluation_response(raw, total_points=total_points)
            results[result_key]["evaluations"].append({
                "response_index": i,
                "verdict": parsed["verdict"],
                "passed_points": parsed["passed_points"],
                "failed_points": parsed["failed_points"],
                "pass_count": parsed["pass_count"],
                "fail_count": parsed["fail_count"],
                "total_points": total_points,
                "raw_response": raw.strip(),
            })
            completed += 1
        except Exception as exc:
            logger.error("[%s] FAILED %s response %d: %s",
                         log_prefix, result_key, i + 1, exc)
            results[result_key]["evaluations"].append({
                "response_index": i,
                "verdict": None,
                "passed_points": [],
                "failed_points": [],
                "pass_count": 0,
                "fail_count": 0,
                "total_points": total_points,
                "error": str(exc),
            })

        save_fn(results)

    return completed, total


def finalize_summaries(results, save_fn):
    for data in results.values():
        data["summary"] = compute_summary(data["evaluations"])
    save_fn(results)


def merge_model_files(prefix, output_path):
    """Combine per-model JSON files into one."""
    combined = {}
    for model_key in MODELS:
        path = _model_file(prefix, model_key)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                model_data = json.load(f)
            combined.update(model_data)
            logger.info("Merged %d entries from %s", len(model_data), model_key)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    logger.info("Combined results → %s (%d entries)", output_path, len(combined))


def load_inputs():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}
    with open(ALG1_RESPONSES_FILE, encoding="utf-8") as f:
        alg1_responses = json.load(f)
    return questions, alg1_responses


def run_parallel(worker_fn, label):
    """Run worker_fn for every model in parallel."""
    questions, alg1_responses = load_inputs()
    logger.info("Starting parallel %s for %d models...", label, len(MODELS))

    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {
            executor.submit(worker_fn, mk, questions, alg1_responses): mk
            for mk in MODELS
        }
        for future in as_completed(futures):
            model_key = futures[future]
            try:
                mk, completed, total = future.result()
                logger.info("Model %s finished: %d/%d", mk, completed, total)
            except Exception as exc:
                logger.error("Model %s raised an exception: %s", model_key, exc)