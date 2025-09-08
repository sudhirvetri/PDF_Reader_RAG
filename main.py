import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
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
class SimpleTfidfEmbeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.fitted = False
    def embed_documents(self, docs):
        texts = [d.page_content if hasattr(d, 'page_content') else str(d) for d in docs]
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        arr = self.vectorizer.transform(texts).toarray()
        return [vec.tolist() for vec in arr]
    def embed_query(self, query):
        arr = self.vectorizer.transform([query]).toarray()
        return arr[0].tolist()

embeddings = SimpleTfidfEmbeddings()
db = Chroma.from_documents(chunks, embedding=embeddings)

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Build RAG pipeline
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a question
while True:
    query = input("\n   Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print(f"\nAnswer: {answer}")
