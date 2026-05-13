# Jaccard Similarity |A ∩ B| / |A ∪ B|
# Inspired by https://codepointtech.com/master-jaccard-similarity-in-python-a-practical-guide/
def jaccard_similarity(response_1, response_2) -> float:
    set_1 = set(response_1.lower().split())
    set_2 = set(response_2.lower().split())

    intersection = len(set_1.intersection(set_2))
    union = len(set_1.union(set_2))

    if union == 0:
        return 0.0
    else:
        return (intersection / union) * 100