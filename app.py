"""
app.py
------
Streamlit CHAT app for the hospital policy knowledge base.

Loads the FAISS index built by ingest.py, retrieves the most relevant
chunks for each user message, and asks Claude to answer using only that
retrieved context — citing department and source file. Keeps full chat
history in the session so you can ask follow-up questions.

Run:
    streamlit run app.py

Requires an ANTHROPIC_API_KEY environment variable (or entered in the
sidebar at runtime) to generate answers. Retrieval still works without a
key — the app will show source passages, just without a generated reply.
"""

import os
import pickle

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

INDEX_DIR = "faiss_index"
DEFAULT_TOP_K = 5

st.set_page_config(page_title="Hospital Policy Assistant", page_icon="🏥", layout="wide")


# ---------------- Cached resources (loaded once per session) ----------------

@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_index_and_metadata():
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    meta_path = os.path.join(INDEX_DIR, "metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None, None

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    return index, meta["chunks"], meta["embedding_model"]


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)


# ---------------- Retrieval + generation ----------------

def retrieve(query: str, index, chunks, model, top_k: int, department_filter: str | None):
    query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    search_k = top_k * 5 if department_filter and department_filter != "All" else top_k
    scores, indices = index.search(query_vec, min(search_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if department_filter and department_filter != "All" and chunk["department"] != department_filter:
            continue
        results.append({**chunk, "score": float(score)})
        if len(results) >= top_k:
            break

    return results


def build_prompt(query: str, retrieved_chunks: list, chat_history: list):
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[Source {i} | Department: {c['department']} | File: {c['source_file']} | Page: {c['page_number']}]\n{c['text']}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant passages found)"

    # Include a short window of prior turns so follow-up questions have context
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 exchanges
        lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent]
        history_text = "Previous conversation:\n" + "\n".join(lines) + "\n\n"

    prompt = f"""You are a hospital policy assistant. Answer the question using ONLY the
context passages below. If the context does not contain the answer, say so
clearly instead of guessing. When you use information from a passage, cite
it inline like [Source N].

{history_text}Context:
{context}

Question: {query}

Answer:"""
    return prompt


def generate_answer(prompt: str, api_key: str):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------- Sidebar ----------------

st.title("🏥 Hospital Policy Knowledge Base Assistant")
st.caption("Ask questions about admissions, department protocols, emergency procedures, and patient safety policies.")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Needed to generate answers. Retrieval-only mode works without it.",
    )
    top_k = st.slider("Source passages per question", min_value=1, max_value=10, value=DEFAULT_TOP_K)

index, chunks, embedding_model_name = load_index_and_metadata()

if index is None:
    st.error(
        "No FAISS index found. Run `python ingest.py` first to build the "
        "knowledge base index from your `hospital_knowledge_base/` folder."
    )
    st.stop()

departments = sorted(set(c["department"] for c in chunks))
with st.sidebar:
    department_filter = st.selectbox("Filter by department", ["All"] + departments)
    st.markdown(f"**Indexed chunks:** {len(chunks)}")
    st.markdown(f"**Departments:** {', '.join(departments)}")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

model = load_embedding_model(embedding_model_name)


# ---------------- Chat state ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str, "sources": [...] (assistant only)}

# Replay existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, r in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Source {i}: {r['source_file']}** — {r['department']} "
                        f"(page {r['page_number']}, score {r['score']:.3f})"
                    )
                    st.write(r["text"])
                    st.divider()

# ---------------- Chat input ----------------

user_input = st.chat_input("Ask a question about hospital policy...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            results = retrieve(user_input, index, chunks, model, top_k, department_filter)

        if not results:
            answer = "I couldn't find any relevant passages for that question. Try rephrasing it or removing the department filter."
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
        elif not api_key_input:
            answer = "Enter an Anthropic API key in the sidebar to get a generated answer. Showing the most relevant source passages instead."
            st.write(answer)
            with st.expander(f"Sources ({len(results)})"):
                for i, r in enumerate(results, 1):
                    st.markdown(
                        f"**Source {i}: {r['source_file']}** — {r['department']} "
                        f"(page {r['page_number']}, score {r['score']:.3f})"
                    )
                    st.write(r["text"])
                    st.divider()
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
        else:
            with st.spinner("Generating answer..."):
                try:
                    prompt = build_prompt(user_input, results, st.session_state.messages[:-1])
                    answer = generate_answer(prompt, api_key_input)
                except Exception as e:
                    answer = f"Could not generate an answer: {e}"

            st.write(answer)
            with st.expander(f"Sources ({len(results)})"):
                for i, r in enumerate(results, 1):
                    st.markdown(
                        f"**Source {i}: {r['source_file']}** — {r['department']} "
                        f"(page {r['page_number']}, score {r['score']:.3f})"
                    )
                    st.write(r["text"])
                    st.divider()

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
