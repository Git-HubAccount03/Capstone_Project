from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL


class LocalLLM:

    def __init__(self):

        print("Loading Ollama Model...")

        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0
        )

        print("Ollama Loaded Successfully.")

    def generate_answer(self, question, context):

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the given context.

If the answer is not available, say:
"I couldn't find the answer in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content