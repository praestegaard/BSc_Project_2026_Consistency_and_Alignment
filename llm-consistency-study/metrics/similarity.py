import logging
from metrics.cosine_similarity import cosine_similarity
from metrics.jaccard_similarity import jaccard_similarity
from metrics.levenshtein_ratio import levenshtein_ratio
from metrics.pubMedBert import pubMedBert_similarity
from metrics.sequence_matcher import sequence_matcher

logger = logging.getLogger(__name__)

def compute_all(response_1: str, response_2: str) -> dict:
    #Compute all metrics
    scores = {
        "jaccard": jaccard_similarity(response_1, response_2),
        "cosine": cosine_similarity(response_1, response_2),
        "sequence_matcher": sequence_matcher(response_1, response_2),
        "levenshtein": levenshtein_ratio(response_1, response_2),
        "pubMedBert": pubMedBert_similarity(response_1, response_2)
    }
    return scores

"""
Simple test
# Consistent responses (same meaning, different wording)
resp1 = "Ibuprofen is a nonsteroidal anti-inflammatory drug used to treat pain and inflammation"
resp2 = "Ibuprofen belongs to the NSAID class and is prescribed for pain relief and reducing inflammation"

# Inconsistent responses (different answers to same question)
resp3 = "Ibuprofen is a nonsteroidal anti-inflammatory drug used to treat pain and inflammation"
resp4 = "Ibuprofen is an antibiotic commonly used to treat bacterial infections"

score1 = compute_all(resp1, resp2)
score2 = compute_all(resp1, resp3)
score3 = compute_all(resp1, resp4)
score4 = compute_all(resp2, resp3)
score5 = compute_all(resp2, resp4)
score6 = compute_all(resp3, resp4)

print(score1)
print(score2)
print(score3)
print(score4)
print(score5)
print(score6)
"""