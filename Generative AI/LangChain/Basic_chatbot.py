from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
load_dotenv()

    
model=ChatGroq(model="openai/gpt-oss-20b", temperature=1)



# response = model.invoke("Tell me a joke about AI overtaking humans")

print("--------- Welcome to the Chat, type 0 to exit ---------")

messages = [
    SystemMessage(content="You are a funny AI agent")
    # Stores previous messages to maintain conversation context.
    # However, sending the entire conversation repeatedly uses more tokens.
    # A very long conversation may exceed the model's context window.
]

while True: #so it does not stop after a single response

    prompt=input("you: ")
    messages.append(HumanMessage(content=prompt))
    if prompt=="0":
        break

    response=model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("Bot: ", response.content)



print(messages)

