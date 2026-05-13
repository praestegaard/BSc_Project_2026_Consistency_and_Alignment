"""
Algorithm 2. Reference-Based Self-Evaluation

Each model evaluates its own K responses against the physician answer.
"""

import logging

from config import ALG2_RESULTS_FILE
from evaluation_runner import (
    count_must_have_points,
    load_results, save_results,
    evaluate_responses, finalize_summaries,
    merge_model_files, run_parallel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)
 
PREFIX = "algorithm2"
 
 
def run_model(model_key, questions, alg1_responses):
    results = load_results(PREFIX, model_key)
    save_fn = lambda r: save_results(PREFIX, model_key, r)
    completed = total = 0
 
    for key, data in alg1_responses.items():
        if data["model"] != model_key:
            continue
 
        qid = data["question_id"]
        physician_answer = questions[qid]["physician_answer"]
        must_have = questions[qid].get("must_have_statements", "")
        total_points = count_must_have_points(must_have)
        responses = [r for r in data["responses"] if r is not None]
 
        c, t = evaluate_responses(
            result_key=key,
            results=results,
            evaluator_model=model_key,
            responses=responses,
            question_text=data["question"],
            physician_answer=physician_answer,
            must_have=must_have,
            total_points=total_points,
            entry_metadata={
                "question_id": qid,
                "question_type": data["question_type"],
                "model": model_key,
                "model_display": data["model_display"],
                "evaluator": model_key,
            },
            save_fn=save_fn,
            log_prefix=model_key,
        )
        completed += c
        total += t
 
    finalize_summaries(results, save_fn)
    logger.info("[%s] Done. %d/%d evaluations completed.", model_key, completed, total)
    return model_key, completed, total
 
 
def run():
    run_parallel(run_model, "self-evaluation")
    merge_model_files(PREFIX, ALG2_RESULTS_FILE)
    logger.info("Algorithm 2 complete.")
 
 
if __name__ == "__main__":
    run()