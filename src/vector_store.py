db = VectorStore()

db.add(
    vectors
)

db.save()
results = db.search(
    query_vector
)