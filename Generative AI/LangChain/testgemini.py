from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

response = model.invoke("tell me a simple joke.")

print(response.content[0]["text"])