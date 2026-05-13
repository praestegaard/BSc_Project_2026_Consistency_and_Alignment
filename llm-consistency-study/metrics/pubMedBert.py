from sentence_transformers import SentenceTransformer
import numpy as np

# NeuML/pubmedbert-base-embeddings, pretrained on PubMed abstracts, should handle medical terminology better than a general model.
# https://huggingface.co/NeuML/pubmedbert-base-embeddings

model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

def pubMedBert_similarity(response_1: str, response_2: str) -> float:
    """Embedding cosine similarity via PubMedBERT, scaled to 0-100."""
    if not response_1.strip() or not response_2.strip():
        return 0.0

    embedding_1 = model.encode(response_1)
    embedding_2 = model.encode(response_2)

    norm_1 = np.linalg.norm(embedding_1)
    norm_2 = np.linalg.norm(embedding_2)
    if norm_1 == 0 or norm_2 == 0:
        return 0.0

    cos_sim = np.dot(embedding_1, embedding_2) / (norm_1 * norm_2)

    # Ensure negative values are set to 0 and scale to 0-100
    return float(max(0.0, cos_sim)) * 100.0