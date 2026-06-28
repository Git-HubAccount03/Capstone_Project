import streamlit as st

from rag_pipeline import RAGPipeline

# -------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Personal Document Chat Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Personal Document Chat Assistant")

st.markdown(
    "Ask questions about your uploaded documents using AI."
)

# -------------------------------------------------------
# Load RAG Pipeline
# -------------------------------------------------------

@st.cache_resource
def load_pipeline():

    return RAGPipeline()


rag = load_pipeline()

# -------------------------------------------------------
# Chat History
# -------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

# Display Chat History

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# -------------------------------------------------------
# Chat Input
# -------------------------------------------------------

question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching Documents..."):

            try:

                answer, docs = rag.ask(question)

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

                with st.expander("📚 Retrieved Source Chunks"):

                    if len(docs) == 0:

                        st.warning("No relevant document found.")

                    else:

                        for i, doc in enumerate(docs, start=1):

                            st.markdown(f"### Chunk {i}")

                            st.write(doc.page_content)

                            if "source" in doc.metadata:

                                st.caption(
                                    f"Source : {doc.metadata['source']}"
                                )

                            if "page" in doc.metadata:

                                st.caption(
                                    f"Page : {doc.metadata['page'] + 1}"
                                )

                            st.divider()

            except Exception as ex:

                st.error(ex)