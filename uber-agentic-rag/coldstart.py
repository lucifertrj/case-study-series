from fastembed import TextEmbedding, SparseTextEmbedding

dense_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-small-en")
sparse_model = SparseTextEmbedding(model_name="Qdrant/BM25")

print(dense_model.embed("Initial loading test"))
print(sparse_model.embed("Initial loading test"))