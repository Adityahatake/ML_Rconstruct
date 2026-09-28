from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

#using the model locally by installing it : pip install langchain-huggingface sentence-transformers numpy
# Free, runs locally on CPU, no API key needed (downloads ~90 MB the first time)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "I love playing football on weekends.",
    "Soccer is my favourite sport to play on Sundays.",
    "The stock market fell sharply today.",
]

# Embed the 3 lines
vectors = np.array(embeddings.embed_documents(texts))
print("Number of vectors:", vectors.shape[0])
print("Dimensions per vector:", vectors.shape[1])
print("First 5 values of line 1:", vectors[0][:5])


# Cosine similarity: closer to 1 = more similar meaning
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("\nSimilarity scores:")
print("Line 1 vs Line 2 (similar meaning):  ", round(cosine(vectors[0], vectors[1]), 3))
print("Line 1 vs Line 3 (different meaning):", round(cosine(vectors[0], vectors[2]), 3))

