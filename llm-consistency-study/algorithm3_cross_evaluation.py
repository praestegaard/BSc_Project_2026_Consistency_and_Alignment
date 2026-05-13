"""
Algorithm 3. Reference-Based Cross-Evaluation

Each model's K responses are evaluated by every OTHER model
against the physician answer.
"""

import logging

from config import MODELS, ALG3_RESULTS_FILE
from evaluation_runner import (
    count_must_have_points,
    load_results, save_results,
    evaluate_responses, finalize_summaries,
    merge_model_files, run_parallel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PREFIX = "algorithm3"


def run_source_model(source_model, questions, alg1_responses):
    results = load_results(PREFIX, source_model)
    save_fn = lambda r: save_results(PREFIX, source_model, r)
    completed = total = 0

    for key, data in alg1_responses.items():
        if data["model"] != source_model:
            continue

        qid = data["question_id"]
        physician_answer = questions[qid]["physician_answer"]
        must_have = questions[qid].get("must_have_statements", "")
        total_points = count_must_have_points(must_have)
        responses = [r for r in data["responses"] if r is not None]

        for evaluator_model in MODELS:
            if evaluator_model == source_model:
                continue

            rkey = f"q{qid}_{source_model}_by_{evaluator_model}"

            c, t = evaluate_responses(
                result_key=rkey,
                results=results,
                evaluator_model=evaluator_model,
                responses=responses,
                question_text=data["question"],
                physician_answer=physician_answer,
                must_have=must_have,
                total_points=total_points,
                entry_metadata={
                    "question_id": qid,
                    "question_type": data["question_type"],
                    "source_model": source_model,
                    "source_model_display": data["model_display"],
                    "evaluator_model": evaluator_model,
                    "evaluator_model_display": MODELS[evaluator_model]["display_name"],
                },
                save_fn=save_fn,
                log_prefix=source_model,
            )
            completed += c
            total += t

    finalize_summaries(results, save_fn)
    logger.info("[%s] Done. %d/%d cross-evaluations completed.", source_model, completed, total)
    return source_model, completed, total


def run():
    run_parallel(run_source_model, "cross-evaluation")
    merge_model_files(PREFIX, ALG3_RESULTS_FILE)
    logger.info("Algorithm 3 complete.")


if __name__ == "__main__":
    run()