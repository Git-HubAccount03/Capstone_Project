from retrieval import DocumentRetriever
from llm import LocalLLM


class RAGPipeline:

    def __init__(self):

        self.retriever = DocumentRetriever()
        self.llm = LocalLLM()

    def ask(self, question):

        docs = self.retriever.retrieve_documents(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        answer = self.llm.generate_answer(
            question,
            context
        )

        return answer, docs