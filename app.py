from flask import Flask, render_template, request, jsonify, session, g
import os
import re
import math
import asyncio

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from werkzeug.middleware.proxy_fix import ProxyFix
from langchain.schema import Document

# -----------------------------
# Flask & Config
# -----------------------------
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.wsgi_app = ProxyFix(app.wsgi_app)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

PDF_PATH = "XID Creation-Modification FAQs - Updated.pdf"
VSTORE_PATH = "xid_faqs_vectorstore_enhanced_v2"  # bumped version to avoid old cache

# -----------------------------
# Event loop helper
# -----------------------------
def ensure_event_loop():
    """Ensure every (waitress) thread has an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# -----------------------------
# Models (cached in app context)
# -----------------------------
def get_embedding_model():
    ensure_event_loop()
    if 'embedding_model' not in g:
        g.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return g.embedding_model

def get_chat_model(temp=0.1, tokens=1000):
    ensure_event_loop()
    # No global cache on temp/tokens variants; the two call-sites use same values.
    return ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=temp, max_tokens=tokens)

# -----------------------------
# Load / Build VectorStore
# -----------------------------
def load_vectorstore():
    """
    Loads FAISS if present; else builds from the PDF with good chunking for FAQs.
    """
    ensure_event_loop()
    embedding_model = get_embedding_model()

    if os.path.exists(VSTORE_PATH):
        return FAISS.load_local(VSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)

    # Build from PDF
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Knowledge base PDF not found at: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )
    docs = splitter.split_documents(pages)

    # Helpful metadata
    for i, doc in enumerate(docs):
        doc.metadata['chunk_id'] = i
        lc = doc.page_content.lower()
        doc.metadata['is_faq_like'] = any(k in lc for k in ['question:', 'q:', 'faq', '?'])

    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(VSTORE_PATH)
    return db

# Build a safe list of all docs for BM25
def all_documents_from_faiss(db: FAISS):
    docs = []
    # db.index.ntotal can be 0 if nothing; guard it
    total = getattr(db.index, "ntotal", 0)
    for i in range(total):
        doc_id = db.index_to_docstore_id.get(i)
        if doc_id is None:
            continue
        doc = db.docstore.search(doc_id)
        if isinstance(doc, Document) and doc.page_content:
            docs.append(doc)
    return docs

# -----------------------------
# Hybrid retriever
# -----------------------------
def get_ensemble_retriever():
    ensure_event_loop()
    if 'retriever' not in g:
        db = load_vectorstore()

        # 1) Semantic retriever: go pure similarity (higher recall for FAQs)
        vector_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # prioritize closest semantic matches
        )

        # 2) Lexical BM25 (for exact phrasing)
        all_docs = all_documents_from_faiss(db)
        bm25 = BM25Retriever.from_documents(all_docs) if all_docs else None
        if bm25:
            bm25.k = 10
            g.retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.7, 0.3])
        else:
            # fallback to vector only
            g.retriever = vector_retriever
    return g.retriever

# -----------------------------
# Prompt & LLM Chain
# -----------------------------
def get_llm_chain():
    if 'llm_chain' not in g:
        llm = get_chat_model(temp=0.1, tokens=1000)
        prompt = PromptTemplate.from_template("""
XID FAQ Assistant Prompt
You are an expert XID FAQ assistant named 'Xid-FAQ'. Your sole purpose is to answer user queries with 100% accuracy based exclusively on the provided knowledge base. You must not use any external information, personal opinions, or general knowledge beyond what is explicitly provided.
CORE BEHAVIOR

Greet users when they greet you
Answer questions using ONLY information from the provided document
Maintain a helpful, professional tone
Provide accurate, specific responses

ANALYSIS INSTRUCTIONS

Direct Match Check: First, identify if the question directly matches any FAQ in the context
Relevance Search: If no direct match, find the most relevant information that addresses the user's concern
Actionable Information: Provide specific, actionable information when available
Clear Formatting: Structure your response with proper formatting
Email References: If a user asks for contact information, refer specifically to email addresses provided in the document

RESPONSE GUIDELINES
DO:

Answer directly without unnecessary preambles
Use bullet points or numbered lists for multi-step processes
Break down complex information into digestible parts
Include only information directly relevant to the question
State what you know and acknowledge when information is partial

AVOID:

Phrases like "The FAQs do not specify" or "As per the FAQs"
Generic responses when specific information is available
Mixing multiple unrelated topics in one answer
Referencing FAQ numbers or sections
Conversational fluff or extraneous details

STRICT CONSTRAINTS

No External Knowledge: Use ONLY the provided document content
No Hallucination: If information cannot be found in the knowledge base, respond with: "I apologize, but I cannot find that information in my knowledge base."
No Extrapolation: Do not make assumptions or infer information not explicitly stated
Direct Answers: Provide factual, concise responses

INPUT FORMAT
Previous Conversation:
{chat_history}

Relevant Context:
{context}

User Question:
{question}
OUTPUT FORMAT
Provide a focused answer based solely on the context provided, following all guidelines above.""")
        g.llm_chain = LLMChain(llm=llm, prompt=prompt)
    return g.llm_chain

# -----------------------------
# Utilities
# -----------------------------
def preprocess_question(q: str) -> str:
    """Light normalization + common expansions (keeps domain terms intact)."""
    q = re.sub(r'\s+', ' ', q.strip())
    # expand a few fragile short forms
    replacements = {
        r'\bxid\b': 'XID',
        r'\bid\b': 'identification',
        r'\bdoc\b': 'document',
        r'\breq\b': 'request',
        r'\bapp\b': 'application'
    }
    for pat, repl in replacements.items():
        q = re.sub(pat, repl, q, flags=re.IGNORECASE)
    return q

def format_history(msgs):
    if not msgs:
        return ""
    recent = msgs[-8:] if len(msgs) > 8 else msgs
    hist = []
    # pair user/assistant
    buf = []
    for m in recent:
        if m.get("role") == "user":
            buf = [m.get("content", "")]
        elif m.get("role") == "assistant" and buf:
            hist.append(f"Previous Q: {buf[0]}\nPrevious A: {m.get('content','')[:200]}...")
            buf = []
    return "\n".join(hist)

def cosine(a, b):
    """Cosine similarity for two python lists of floats."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# -----------------------------
# Query Expansion (LLM-assisted)
# -----------------------------
def expand_query_variants(original_q: str, max_variants: int = 3):
    """
    Uses the chat model to produce a few paraphrases to boost recall on hybrid retrieval.
    Returns [original_q, var1, var2, ...].
    """
    try:
        llm = get_chat_model(temp=0.3, tokens=200)
        prompt = (
            "Rewrite the following user question into up to 3 semantically equivalent variations "
            "that might use different wording but keep the same intent. "
            "Do not add new facts.\n\n"
            f"Question: {original_q}\n"
            "Variations:"
        )
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", "") or ""
        # split into lines, strip bullets/prefixes
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        cleaned = []
        for l in lines:
            l = re.sub(r"^[\-\*\d\.\)]\s*", "", l)  # remove bullet/number prefixes
            if l and l.lower() != original_q.lower():
                cleaned.append(l)
            if len(cleaned) >= max_variants:
                break
        return [original_q] + cleaned
    except Exception:
        # On any issue, just use the original query
        return [original_q]

# -----------------------------
# Retrieval + Re-ranking
# -----------------------------
def retrieve_candidates(query: str, k_each: int = 8, dedup: bool = True):
    """
    Runs retrieval on multiple query variants and merges results.
    """
    retriever = get_ensemble_retriever()
    variants = expand_query_variants(query, max_variants=3)

    seen_ids = set()
    merged = []
    for v in variants:
        try:
            docs = retriever.get_relevant_documents(v)
        except AttributeError:
            # if retriever is a plain BaseRetriever
            docs = retriever.invoke(v)

        for d in docs[:k_each]:
            # Create a stable key for dedup (content + page)
            key = (d.page_content, d.metadata.get("source"), d.metadata.get("page"))
            if not dedup or key not in seen_ids:
                seen_ids.add(key)
                merged.append(d)

    # Limit to a reasonable pool for reranking
    return merged[:30] if len(merged) > 30 else merged

def rerank_by_similarity(query: str, docs):
    """
    Compute cosine similarity between query and each doc using the embedding model, then sort.
    """
    if not docs:
        return []

    embedding_model = get_embedding_model()
    q_vec = embedding_model.embed_query(query)

    # embed_documents can batch; here do simple loop for clarity
    scored = []
    for d in docs:
        try:
            d_vec = embedding_model.embed_documents([d.page_content])[0]
            sim = cosine(q_vec, d_vec)
        except Exception:
            sim = 0.0
        scored.append((d, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# -----------------------------
# Core Answering
# -----------------------------
SIMILARITY_MIN_THRESHOLD = 0.60   # below this => apologize fallback
TOP_CONTEXT_K = 5

def get_answer(user_q: str, chat_history: str):
    ensure_event_loop()
    q = preprocess_question(user_q)

    # 1) Retrieve a pooled set of candidates (hybrid + query variants)
    candidates = retrieve_candidates(q, k_each=8, dedup=True)

    # 2) Re-rank by cosine similarity on the SAME embedding space as FAISS
    ranked = rerank_by_similarity(q, candidates)

    # 3) If nothing or weak match, return fallback
    if not ranked or ranked[0][1] < SIMILARITY_MIN_THRESHOLD:
        return "I apologize, but I cannot find that information in my knowledge base.", []

    # 4) Take top K docs into context
    top_docs = [d for d, _ in ranked[:TOP_CONTEXT_K]]
    context = "\n\n".join([d.page_content for d in top_docs])

    # 5) Generate answer with your strict prompt
    llm_chain = get_llm_chain()
    answer = llm_chain.run(context=context, question=q, chat_history=chat_history)
    return answer, top_docs

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    if 'messages' not in session:
        session['messages'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    ensure_event_loop()
    try:
        user_message = request.get_json().get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        session.setdefault('messages', []).append({"role": "user", "content": user_message})
        chat_history = format_history(session['messages'][:-1])

        answer, _ = get_answer(user_message, chat_history)

        session['messages'].append({"role": "assistant", "content": answer})
        session.modified = True
        return jsonify({'response': answer, 'success': True})
    except Exception as e:
        print("Chat error:", e)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    session['messages'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
