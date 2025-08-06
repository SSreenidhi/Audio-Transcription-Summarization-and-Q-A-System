def get_relevant_chunks(index, embedding_model, question, chunks, k=3):
    question_embedding = embedding_model.encode([question]).astype("float32")
    D, I = index.search(question_embedding, k=k)
    return [chunks[i] for i in I[0]]