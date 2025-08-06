import numpy as np

def embed_chunks(embedding_model, chunks):
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    return np.array(embeddings).astype("float32")