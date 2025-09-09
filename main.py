import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load PDF documents
pdf_folder = "data"
documents = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, filename))
        documents.extend(loader.load())

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed chunks using TF-IDF (local, no download required)
# (Commented out — replaced by semantic embeddings below)
# class SimpleTfidfEmbeddings:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer()
#         self.fitted = False
#     def embed_documents(self, docs):
#         texts = [d.page_content if hasattr(d, 'page_content') else str(d) for d in docs]
#         if not self.fitted:
#             self.vectorizer.fit(texts)
#             self.fitted = True
#         arr = self.vectorizer.transform(texts).toarray()
#         return [vec.tolist() for vec in arr]
#     def embed_query(self, query):
#         arr = self.vectorizer.transform([query]).toarray()
#         return arr[0].tolist()

# Semantic embeddings using sentence-transformers/all-MiniLM-L6-v2
class SentenceTransformersEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # loads the model (will download the model the first time)
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, docs):
        texts = [d.page_content if hasattr(d, 'page_content') else str(d) for d in docs]
        arr = self.model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in arr]
    def embed_query(self, query):
        arr = self.model.encode([query], convert_to_numpy=True)
        return arr[0].tolist()

embeddings = SentenceTransformersEmbeddings()
db = Chroma.from_documents(chunks, embedding=embeddings)

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Build RAG pipeline
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a question (interactive) or run a single test query when TEST_QUERY is set
test_query = os.getenv("TEST_QUERY")
if test_query:
    print("Running TEST_QUERY...")
    answer = qa_chain.run(test_query)
    print(f"\nAnswer: {answer}")
else:
    while True:
        query = input("\n   Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = qa_chain.run(query)
        print(f"\nAnswer: {answer}")
