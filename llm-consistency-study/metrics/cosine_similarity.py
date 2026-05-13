# Cosine Similarity (using TF-IDF)
# Inspired by https://coderivers.org/blog/cosine-similarity-python/
def cosine_similarity(response_1: str, response_2: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cs_similarity

    if not response_1.strip() or not response_2.strip():
        return 0.0

    responses = [response_1, response_2]

    vectorizer = TfidfVectorizer()
    try:
        response_vectors = vectorizer.fit_transform(responses)
    except ValueError:
        return 0.0

    similarity = cs_similarity(response_vectors)

    # Extract the similarity between response 1 and response 2
    return float(similarity[0][1]) * 100