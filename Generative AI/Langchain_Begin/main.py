from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-latest"
)

response = model.invoke("Tell me a joke about AI")

print(response.content)