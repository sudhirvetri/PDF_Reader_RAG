# Changelog

All notable changes made to the codebase during the current session.

## [Unreleased]
- Added `data/sample.pdf` — a small test PDF used for verifying the RAG pipeline.
- Installed missing dependency `langchain-community` required for `PyPDFLoader`.
- Added `SentenceTransformersEmbeddings` class using `sentence-transformers/all-MiniLM-L6-v2`.
- Commented out the original TF-IDF embedding class and replaced its instantiation with the new semantic embeddings class (kept TF-IDF implementation in comments for reference).
- Added a `TEST_QUERY` environment variable path in `main.py` to support non-interactive testing.
- Created this `CHANGELOG.md` documenting the above changes.


