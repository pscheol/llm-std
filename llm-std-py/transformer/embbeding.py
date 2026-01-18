from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

vector = model.encode("Best movice ever!")

vector.shape