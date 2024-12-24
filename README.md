# 📚 Summarize & Chat Hub

**An interactive AI-powered application for summarizing and exploring YouTube videos and web content.**

## 🚀 **Overview**
Summarize & Chat Hub is a user-friendly Streamlit application designed to:
- **Summarize YouTube videos and web pages** quickly and effectively.
- **Provide interactive Q&A** sessions based on the content.
- **Support multilingual transcripts** (English & Hindi).

Powered by **LangChain**, **Google Gemini API**, and **yt_dlp**, this tool transforms lengthy video transcripts and web content into concise summaries and enables interactive question-answering.

## 🛠️ **Key Features**
- 📺 Summarize YouTube videos with metadata extraction.
- 🌐 Analyze and summarize web page content.
- 🗨️ Chat with the summarized content seamlessly.
- 🎯 Multilingual transcript support.
- ⚡ Real-time Q&A powered by Gemini LLM.

## 🧩 **Tech Stack**
- **Frameworks:** Streamlit, LangChain
- **LLM:** Google Gemini API
- **Embeddings:** Google GenerativeAIEmbeddings
- **Data Loaders:** yt_dlp, UnstructuredURLLoader
- **Vector Store:** FAISS
- **Deployment:** AWS

## 💻 **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/aqib0770/Content-Summarization-QA.git
   cd Content-Summarization-QA
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API key:
   - Get your **Google Gemini API key** from [here](https://aistudio.google.com/prompts/new_chat).
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🔑 **Environment Variables**
- `GEMINI_API_KEY`: Your Google Gemini API key.

## 📦 **How It Works**
1. Enter a **YouTube** or **Web URL**.
2. Click **Summarize** to generate insights.
3. Chat with the content using the interactive chat interface.

## 🧑‍💻 **Contribution**
Contributions are welcome! Feel free to open an issue or submit a pull request.


## 🌟 **Acknowledgment**
Special mention for the [langchain_yt_dlp](https://pypi.org/project/langchain-yt-dlp/) integration, a package I contributed to LangChain for enhanced YouTube content handling.

## 🔗 **Live Demo**
[Access the deployed application here](https://summarizechat.streamlit.app/)

---
Feel free to share your feedback and suggestions!

