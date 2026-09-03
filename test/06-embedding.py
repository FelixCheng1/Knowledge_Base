from app.lm.embedding_utils import generate_embeddings

texts = ["Hello,world","你好，世界"]
print(generate_embeddings(texts))