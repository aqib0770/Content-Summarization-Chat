import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from langchain_core.output_parsers import StrOutputParser


st.set_page_config(page_title="Summarize & Chat Hub")
st.title("Summarize and Chat with YouTube & Web Content")
st.subheader("Enter the URL")
st.write("Please enter a URL with proper content or transcript to summarize and ask questions")
url = st.text_input("URL", label_visibility="collapsed")

with st.sidebar:
    google_api_key = st.text_input("Enter Gemini API key", value="", type="password")
    st.markdown("Get your API key from [here](https://aistudio.google.com/prompts/new_chat)")
    video_info = st.checkbox("Add video metadata", value=False)
    lang = st.selectbox("Transcript Language", ["en", "hi"], index=0)

os.environ["GROQ_API_KEY"] = google_api_key

def create_llm(api_key):
    """Initialize the Google Generative AI LLM."""
    return ChatGoogleGenerativeAI(api_key=api_key, model="models/gemini-2.0-flash-exp")


summary_prompt = PromptTemplate(
    template="""
    If the content is empty, repetitive or irrelevant, respond with 
    "The content could not be summarized meaningfully."
    Otherwise, provide a summary of the following content:
    Content: {text}
    """,
    input_variables=["text"]
)

qa_prompt = ChatPromptTemplate.from_template(
    """You are an intelligent assistant with expertise in analyzing
    transcripts and providing responses in a conversational, human-like manner.
    You have complete understanding of the provided transcript and can answer
    any question naturally without referencing the source explicitly.
    <context>
    {context}
    Question: {input}
    """
)


def get_loader_obj(url, lang, video_info):
    """Load content from the given URL."""
    metadata = None
    if "youtube.com" in url:
        if video_info:
            loader = YoutubeLoader.from_youtube_url(youtube_url=url, language=lang)
            metadata = YoutubeLoaderDL.from_youtube_url(url, add_video_info=video_info).load()
        else:
            loader = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info=video_info)
    else:
        loader = UnstructuredURLLoader(
            urls=[url],
            ssl_verified=False,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0"}
        )
    docs = loader.load()
    return docs, metadata


def create_vector_embeddings(docs, api_key):
    """Create FAISS vector embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model='models/text-embedding-004')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs)
    return FAISS.from_documents(final_docs, embeddings)


def create_summary_chain(docs, api_key):
    """Create a summarization chain."""
    llm = create_llm(api_key)
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
    return chain.run(docs)


def generate_response(user_input, retriever, api_key):
    """Generate a response using retrieval-based chain."""
    llm = create_llm(api_key)
    document_chain = create_stuff_documents_chain(llm, qa_prompt, output_parser=StrOutputParser())
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    for chunk in retrieval_chain.stream({"input": user_input}):
        yield chunk.get('answer', '')


if "summary" not in st.session_state:
    st.session_state.summary = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "messages" not in st.session_state:
    st.session_state.messages = []


if st.button("Summarize"):
    if not google_api_key.strip() or not url.strip():
        st.error("Please provide the required information")
    elif not validators.url(url):
        st.error("Please enter a valid URL (YouTube or Website)")
    else:
        try:
            with st.spinner("Generating summary..."):
                docs, metadata = get_loader_obj(url, lang, video_info)
                if not docs or all(not doc.page_content.strip() for doc in docs):
                    st.error("No content or transcript found in the provided URL")
                    st.stop()

                st.session_state.docs = docs
                st.session_state.metadata = metadata
                st.session_state.summary = create_summary_chain(docs, google_api_key)

                if "vectors" not in st.session_state or not st.session_state.vectors:
                    with st.spinner("Preparing for Q&A..."):
                        st.session_state.vectors = create_vector_embeddings(docs, google_api_key)
        except Exception as e:
            st.exception(f"Exception: {e}")

if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)

if st.session_state.metadata:
    st.subheader("Video Metadata")
    st.json(st.session_state.metadata[0].metadata, expanded=False)


if st.session_state.vectors:
    st.subheader("Ask Questions")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask a question"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            response_generator = generate_response(
                user_input, st.session_state.vectors.as_retriever(), google_api_key
            )
            full_response = ""
            for chunk in st.write_stream(response_generator):
                full_response += chunk
        st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.button("Clear chat", on_click=lambda: st.session_state.messages.clear())
