from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="phi3:mini",
    temperature=0
)

response = llm.invoke("What is AI?")

print(response.content)