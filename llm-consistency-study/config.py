"""
Config: Models, paths, and study parameters.

Each model entry points to the default free-tier model the platform
serves to consumers (checked April 2026).
"""

MODELS = {
    "chatgpt": {
        "provider": "openai",
        "model_id": "gpt-5.3-chat-latest",
        "display_name": "ChatGPT (GPT-5.3 Instant)",
        "env_key": "OPENAI_API_KEY",
    },
     "claude": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "display_name": "Claude (Sonnet 4.6)",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "mistral": {
        "provider": "mistral",
        "model_id": "mistral-medium-latest",
        "display_name": "Le Chat (Mistral Medium latest)",
        "env_key": "MISTRAL_API_KEY",
    },
    "gemini": {
        "provider": "google",
        "model_id": "gemini-3-flash-preview",
        "display_name": "Gemini (3 Flash)",
        "env_key": "GOOGLE_API_KEY",
    },
}

K = 5                       # repetitions per question per model
PASS_THRESHOLD = 0.80       # 80% alignment needed to pass

DATA_DIR = "data"
RESULTS_DIR = "results"
QUESTIONS_FILE = f"{DATA_DIR}/questions.json"

ALG1_RESPONSES_FILE = f"{RESULTS_DIR}/algorithm1_responses.json"
ALG1_SCORES_FILE = f"{RESULTS_DIR}/algorithm1_similarity_scores.json"
ALG2_RESULTS_FILE = f"{RESULTS_DIR}/algorithm2_self_evaluation.json"
ALG3_RESULTS_FILE = f"{RESULTS_DIR}/algorithm3_cross_evaluation.json"
CROSS_ANALYSIS_FILE = f"{RESULTS_DIR}/cross_algorithm_analysis.json"

MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 20
REQUEST_DELAY_SECONDS = 5   # courtesy delay so we don't hammer the APIs
