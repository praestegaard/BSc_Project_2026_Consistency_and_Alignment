"""
Main runner — runs the full pipeline or individual steps.

  python run_study.py              # all steps
  python run_study.py --step 1     # Algorithm 1 only
  python run_study.py --step 2     # Algorithm 2 only
  python run_study.py --step 3     # Algorithm 3 only
  python run_study.py --step 4     # cross-analysis only, no API calls!
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def check_env():
    """Verify all required API keys are set."""
    from config import MODELS

    missing = []
    for key, cfg in MODELS.items():
        env = cfg["env_key"]
        if not os.environ.get(env):
            missing.append(f"  {env}  (for {cfg['display_name']})")

    if missing:
        logger.error("Missing API keys:\n%s", "\n".join(missing))
        logger.error(
            "\nSet them as environment variables, e.g.:\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "  export GOOGLE_API_KEY='AI...'\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "  export MISTRAL_API_KEY='...'\n"
        )
        sys.exit(1)

    logger.info("All API keys found.")


def check_questions():
    """Verify the questions file exists."""
    from config import QUESTIONS_FILE
    if not os.path.exists(QUESTIONS_FILE):
        logger.error(
            "Questions file not found: %s\n"
            "Create it with your 60 medical questions in JSON format.\n"
            "See data/questions.json for the expected structure.",
            QUESTIONS_FILE,
        )
        sys.exit(1)


def run_step1():
    logger.info("=" * 60)
    logger.info("ALGORITHM 1 — Consistency Analysis")
    logger.info("=" * 60)
    import algorithm1_consistency
    algorithm1_consistency.run()


def run_step2():
    logger.info("=" * 60)
    logger.info("ALGORITHM 2 — Self-evaluation (Reference-Based)")
    logger.info("=" * 60)
    import algorithm2_self_evaluation
    algorithm2_self_evaluation.run()


def run_step3():
    logger.info("=" * 60)
    logger.info("ALGORITHM 3 — Cross-evaluation (Reference-Based)")
    logger.info("=" * 60)
    import algorithm3_cross_evaluation
    algorithm3_cross_evaluation.run()


def run_step4():
    logger.info("=" * 60)
    logger.info("CROSS-ALGORITHM ANALYSIS")
    logger.info("=" * 60)
    import cross_analysis
    cross_analysis.run()



def main():
    parser = argparse.ArgumentParser(description="LLM Medical Consistency Study")
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run a specific step only (1-4). Default: run all.",
    )
    args = parser.parse_args()


    if args.step is None:
        check_env()
        check_questions()
        run_step1()
        run_step2()
        run_step3()
        run_step4()
    elif args.step in (1, 2, 3):
        check_env()
        check_questions()
        if args.step == 1:
            run_step1()
        elif args.step == 2:
            run_step2()
        elif args.step == 3:
            run_step3()
    elif args.step == 4:
        check_questions()
        run_step4()

    logger.info("Study complete.")


if __name__ == "__main__":
    main()