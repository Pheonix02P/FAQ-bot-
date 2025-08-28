# xid_faq_bot_app.py

from flask import Flask, render_template, request, jsonify, session, g
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.retrievers.document_compressors import LLMChainExtractor
import re
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.wsgi_app = ProxyFix(app.wsgi_app)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyDoVdAvZXiHVdzZck30JtUkTXBmbvPJgJU")

# Load once and cache in app context
def get_embedding_model():
    if 'embedding_model' not in g:
        g.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return g.embedding_model

def load_vectorstore():
    file_path = "XID Creation-Modification FAQs - Updated.pdf"
    vectorstore_path = "xid_faqs_vectorstore_enhanced"
    embedding_model = get_embedding_model()

    if os.path.exists(vectorstore_path):
        return FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
        )
        docs = splitter.split_documents(pages)
        for i, doc in enumerate(docs):
            doc.metadata['chunk_id'] = i
            if any(k in doc.page_content.lower() for k in ['question:', 'q:', 'faq', '?']):
                doc.metadata['is_faq'] = True

        db = FAISS.from_documents(docs, embedding_model)
        db.save_local(vectorstore_path)
        return db

def get_ensemble_retriever():
    if 'retriever' not in g:
        db = load_vectorstore()
        vector_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7})
        all_docs = [db.docstore.search(db.index_to_docstore_id[i]) for i in range(db.index.ntotal)]
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = 8
        g.retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.6, 0.4])
    return g.retriever

def get_llm_chain():
    if 'llm_chain' not in g:
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.1, max_tokens=1000)
        prompt = PromptTemplate.from_template("""
You are an expert XID FAQ assistant. Your task is to provide precise, relevant answers based solely on the provided context. Also if someone is greeting you, greet them back
ANALYSIS INSTRUCTIONS:
1. First, identify if the question directly matches any FAQ in the context
2. If no direct match, find the most relevant information that addresses the user's concern
3. Provide specific, actionable information when available
4. Format your response clearly with proper structure
5. If a users input a query of asking a email address, refer to the email addresses given in the doument
RESPONSE GUIDELINES:
- Answer directly without unnecessary preambles
- Use bullet points or numbered lists for multi-step processes
- Break down complex information into digestible parts
- Only include information that's directly relevant to the question
- If information is partially available, state what you know and what might need clarification
AVOID:
- Phrases like "The FAQs do not specify" or "As per the FAQs"
- Generic responses when specific information is available
- Mixing multiple unrelated topics in one answer
- Referencing FAQ numbers or sections

Previous Conversation:
{chat_history}

Relevant Context:
{context}

User Question: {question}

Focused Answer:
""")
        g.llm_chain = LLMChain(llm=llm, prompt=prompt)
    return g.llm_chain

def preprocess_question(q):
    q = re.sub(r'\s+', ' ', q.strip())
    for a, f in {'xid': 'XID', 'id': 'identification', 'doc': 'document', 'req': 'request', 'app': 'application'}.items():
        q = re.sub(rf"\\b{a}\\b", f, q, flags=re.IGNORECASE)
    return q

def format_history(msgs):
    if not msgs: return ""
    recent = msgs[-8:] if len(msgs) > 8 else msgs
    hist = []
    for i in range(0, len(recent), 2):
        if i+1 < len(recent):
            hist.append(f"Previous Q: {recent[i]['content']}\nPrevious A: {recent[i+1]['content'][:200]}...")
    return "\n".join(hist)

def get_answer(q, chat_history):
    retriever = get_ensemble_retriever()
    db = load_vectorstore()
    docs = retriever.invoke(q)
    scored_docs = sorted(
        [(d, sum(0.2 for t in ['xid', 'creation', 'modification', 'request'] if t in q.lower() and t in d.page_content.lower())) for d in docs],
        key=lambda x: x[1], reverse=True
    )
    top_docs = [d for d, _ in scored_docs[:5]]
    context = "\n\n".join([d.page_content for d in top_docs])
    llm_chain = get_llm_chain()
    return llm_chain.run(context=context, question=q, chat_history=chat_history), top_docs

@app.route('/')
def index():
    if 'messages' not in session:
        session['messages'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.get_json().get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        session.setdefault('messages', []).append({"role": "user", "content": user_message})
        chat_history = format_history(session['messages'][:-1])
        answer, _ = get_answer(preprocess_question(user_message), chat_history)
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

if __name__ == "__main__":
    from waitress import serve
    import os
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)

