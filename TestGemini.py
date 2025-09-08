
# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Use the correct wrapper
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Test the model
response = llm.invoke("Hello Gemini, how are you?")
print(response)
