# LLM Medical Consistency Study

Tests whether consumer-facing LLMs give consistent and medically accurate
answers, inspired by the framework from Patwardhan, Vaidya & Kundu (2025).

## Setup

```bash
pip install -r requirements.txt

# set API keys (or copy .env.template → .env and source it)
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export GOOGLE_API_KEY='...'
export MISTRAL_API_KEY='...'
```

Questions go in `data/questions.json` — see format below.

## Running

```bash
python run_study.py              # full pipeline
python run_study.py --step 1     # Algorithm 1: consistency
python run_study.py --step 2     # Algorithm 2: self-evaluation
python run_study.py --step 3     # Algorithm 3: cross-evaluation
python run_study.py --step 4     # cross-analysis spreadsheet (no API calls)
python spearman_analysis.py      # Spearman correlations (no API calls)
```

Everything saves after each API call, interrupted runs can be resumed
by re-running the same command.

## Questions format

`data/questions.json` — array of objects:

```json
{
    "id": 1,
    "type": "informational",
    "question": "What is hypertension and what causes it?",
    "physician_answer": "Hypertension is high blood pressure...",
    "must_have_statements": "1. Hypertension means high blood pressure.\n2. It can increase the risk of cardiovascular disease."
}
```

`type` is either `"informational"` or `"situational"`.
Physician answers come from K-QA: https://github.com/Itaymanes/K-QA
