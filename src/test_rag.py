from rag_pipeline import RAGPipeline

rag = RAGPipeline()

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    answer, context = rag.ask(question)

    print("\nAnswer")
    print("-" * 50)
    print(answer)

    print("\nRetrieved Context")
    print("-" * 50)
    print(context)