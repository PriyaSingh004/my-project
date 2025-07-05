!pip install streamlit pyngrok --quiet
# 📦 Install required libraries
!pip install streamlit pyngrok groq sentence-transformers faiss-cpu transformers PyPDF2 --quiet

# ✅ Optional: unzip ngrok (helps in some Colab cases)
!wget -q -c https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip -o ngrok-stable-linux-amd64.zip
!pip install PyPDF2 --quiet

%%writefile app.py
import streamlit as st
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from transformers import pipeline

# ------- PDF Text Extraction -------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return ' '.join(text.split())

# ------- Document Indexing -------
class DocumentIndexer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def add_documents(self, texts):
        self.documents.extend(texts)
        embeddings = self.model.encode(texts)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, k=3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]

# ------- QA Systems -------
class GroqQA:
    def __init__(self, model="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client = Groq()
        self.model = model

    def answer_question(self, context, question):
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"Answer strictly based on: {context}"},
                    {"role": "user", "content": question}
                ],
                model=self.model,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq API error: {e}"

class HuggingFaceQA:
    def __init__(self):
        self.pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    def answer_question(self, context, question):
        try:
            return self.pipeline(question=question, context=context)['answer']
        except Exception as e:
            return f"HuggingFace error: {e}"

# ------- Streamlit UI -------
st.title("📘 Product Manual Q&A")
st.markdown("Upload a PDF manual and ask questions")

groq_api_key = st.text_input("Enter your Grok API key", type="password")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

uploaded_file = st.file_uploader("Upload PDF Manual", type=["pdf"])

if uploaded_file:
    with open("/content/Galaxy S24 ultra.env.pdf", "wb") as f:
        f.write(uploaded_file.read())
    full_text = extract_text_from_pdf("/content/Galaxy S24 ultra.env.pdf")

    chunk_size = 1000
    overlap = 200
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size - overlap)]

    st.success(f"📄 Extracted and indexed {len(chunks)} chunks.")
    indexer = DocumentIndexer()
    indexer.add_documents(chunks)

    groq_qa = GroqQA()
    hf_qa = HuggingFaceQA()

    question = st.text_input("Ask a question from the manual")

    if st.button("Get Answer") and question:
        relevant = indexer.search(question, k=3)
        context = "\n\n".join(relevant)
        answer = groq_qa.answer_question(context, question)
        if not answer or "couldn't find" in answer.lower():
            answer = hf_qa.answer_question(context, question)
        st.markdown("### 💡 Answer:")
        st.write(answer)
from pyngrok import ngrok, conf

# Paste your Ngrok authtoken here
authtoken = "2zPQswwSYYycnNmMJ6QxZAb09t8_2tBy3X8P5Qjt5h82mcYZw"

# Configure the authtoken
conf.get_default().auth_token = authtoken

from pyngrok import ngrok
import threading
from pyngrok import conf

# Paste your Ngrok authtoken here
authtoken = "2zPTAhSPrzB64Aumtshz7clO273_2BNuuoEjgKBSoAAhH2aBn" # Get the authtoken from cell RLEv-BNE3vrb

# Configure the authtoken
conf.get_default().auth_token = authtoken

# Stop any existing ngrok processes before starting a new one
try:
    ngrok.kill()
except:
    pass

# Start ngrok tunnel
public_url = ngrok.connect(8501, "http")

print(f"🌍 Streamlit app is live at: {public_url}")

# Run Streamlit in a thread
def run_app():
    !streamlit run app.py --server.port 8501

thread = threading.Thread(target=run_app)
thread.start()

!ngrok config add-authtoken 2zJ9amneRNNzvZUwdK6L5PZ2SDQ_2bemTt7von6fwbrq4x4S8