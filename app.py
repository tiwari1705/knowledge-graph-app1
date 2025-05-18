# app.py
import streamlit as st 
import fitz  # PyMuPDF
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        return " ".join(p.get_text() for p in paragraphs)
    except Exception as e:
        st.error(f"Failed to fetch URL content: {e}")
        return ""

def extract_triplets(text):
    doc = nlp(text)
    triplets = []
    for sent in doc.sents:
        subj = ""
        verb = ""
        obj = ""
        for token in sent:
            if "subj" in token.dep_:
                subj = token.text
            if token.pos_ == "VERB":
                verb = token.text
            if "obj" in token.dep_:
                obj = token.text
        if subj and verb and obj:
            triplets.append((subj, verb, obj))
    return triplets

def draw_graph(triplets):
    G = nx.DiGraph()
    for subj, rel, obj in triplets:
        G.add_edge(subj, obj, label=rel)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Knowledge Graph")
    st.pyplot(plt)

# ---- Streamlit UI ----
st.title("📚 Knowledge Graph Generator")
st.write("Upload a `.txt` or `.pdf` file or enter a URL to generate a knowledge graph.")

input_method = st.radio("Choose input method", ["Upload File", "Enter URL"])

text = ""

if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload file", type=["pdf", "txt"])
    if uploaded_file:
        text = extract_text(uploaded_file)

elif input_method == "Enter URL":
    url = st.text_input("Paste URL here")
    if url:
        text = extract_text_from_url(url)

if text:
    st.subheader("Extracted Text")
    st.text_area("Extracted Text", text, height=200)

    if st.button("Generate Knowledge Graph"):
        with st.spinner("Extracting triplets..."):
            triplets = extract_triplets(text)

        if triplets:
            st.success(f"Extracted {len(triplets)} triplets.")
            draw_graph(triplets)

            if st.checkbox("Show Triplets"):
                st.write(triplets)
        else:
            st.warning("No triplets found.")
