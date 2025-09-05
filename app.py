from flask import Flask, render_template, request, jsonify, session, g
import os
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from werkzeug.middleware.proxy_fix import ProxyFix
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.wsgi_app = ProxyFix(app.wsgi_app)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Enhanced query expansion keywords
QUERY_EXPANSION_MAP = {
    'xid': ['XID', 'identification document', 'ID card', 'identity'],
    'create': ['creation', 'new', 'apply', 'generate', 'make', 'issue'],
    'modify': ['modification', 'update', 'change', 'edit', 'correct', 'amend'],
    'process': ['procedure', 'steps', 'how to', 'method', 'workflow'],
    'requirement': ['required', 'need', 'necessary', 'mandatory', 'document'],
    'fee': ['cost', 'charge', 'payment', 'amount', 'price'],
    'time': ['duration', 'period', 'days', 'processing time', 'timeline'],
    'status': ['track', 'check', 'progress', 'update', 'follow up']
}

# Ensure every thread has an event loop
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Load once and cache in app context    
def get_embedding_model():
    ensure_event_loop()
    if 'embedding_model' not in g:
        g.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return g.embedding_model

def get_reranker_model():
    """Load sentence transformer for reranking"""
    if 'reranker' not in g:
        try:
            g.reranker = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            g.reranker = None  # Fallback if sentence-transformers not available
    return g.reranker

def load_vectorstore():
    ensure_event_loop()
    file_path = "XID Creation-Modification FAQs - Updated.pdf"
    vectorstore_path = "xid_faqs_vectorstore_enhanced"
    embedding_model = get_embedding_model()

    if os.path.exists(vectorstore_path):
        return FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Enhanced text splitting with better parameters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Reduced for more focused chunks
            chunk_overlap=100,  # Reduced overlap
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
        )
        docs = splitter.split_documents(pages)
        
        # Enhanced metadata extraction
        for i, doc in enumerate(docs):
            doc.metadata['chunk_id'] = i
            content_lower = doc.page_content.lower()
            
            # Enhanced FAQ detection
            if any(pattern in content_lower for pattern in [
                'question:', 'q:', 'faq', '?', 'how to', 'what is', 'can i', 'do i need'
            ]):
                doc.metadata['is_faq'] = True
                doc.metadata['priority'] = 'high'
            
            # Topic classification
            if any(term in content_lower for term in ['create', 'creation', 'new', 'apply']):
                doc.metadata['topic'] = 'creation'
            elif any(term in content_lower for term in ['modify', 'modification', 'update', 'change']):
                doc.metadata['topic'] = 'modification'
            elif any(term in content_lower for term in ['requirement', 'document', 'need']):
                doc.metadata['topic'] = 'requirements'
            elif any(term in content_lower for term in ['fee', 'cost', 'payment']):
                doc.metadata['topic'] = 'fees'
            else:
                doc.metadata['topic'] = 'general'

        db = FAISS.from_documents(docs, embedding_model)
        db.save_local(vectorstore_path)
        return db

def get_ensemble_retriever():
    ensure_event_loop()
    if 'retriever' not in g:
        db = load_vectorstore()
        # Enhanced vector retrieval with better parameters
        vector_retriever = db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 10, "fetch_k": 25, "lambda_mult": 0.8}
        )
        
        all_docs = [db.docstore.search(db.index_to_docstore_id[i]) for i in range(db.index.ntotal)]
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = 10
        
        # Adjusted weights for better balance
        g.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25], 
            weights=[0.7, 0.3]
        )
    return g.retriever

def get_llm_chain():
    if 'llm_chain' not in g:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash", 
            temperature=0.05,  # Lower temperature for more consistent answers
            max_tokens=1200    # Slightly increased for complete answers
        )
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

IMPORTANT: Focus on the most relevant parts of the context that directly answer the user's question. Prioritize recent context and FAQ-specific information.""")
        g.llm_chain = LLMChain(llm=llm, prompt=prompt)
    return g.llm_chain

def expand_query(query):
    """Enhanced query expansion with better keyword mapping"""
    expanded_terms = []
    query_lower = query.lower()
    
    for key, synonyms in QUERY_EXPANSION_MAP.items():
        if key in query_lower:
            expanded_terms.extend(synonyms)
    
    # Add original query terms
    expanded_terms.extend(query.split())
    
    return ' '.join(set(expanded_terms))

def preprocess_question(q):
    """Enhanced question preprocessing"""
    # Clean whitespace
    q = re.sub(r'\s+', ' ', q.strip())
    
    # Enhanced abbreviation expansion
    abbreviations = {
        r'\bxid\b': 'XID',
        r'\bid\b': 'identification',
        r'\bdoc\b': 'document',
        r'\breq\b': 'request',
        r'\bapp\b': 'application',
        r'\bmod\b': 'modification',
        r'\bcreate\b': 'creation',
        r'\bproc\b': 'process'
    }
    
    for abbr, full in abbreviations.items():
        q = re.sub(abbr, full, q, flags=re.IGNORECASE)
    
    return q

def rerank_documents(query, documents, reranker_model):
    """Enhanced document reranking using semantic similarity"""
    if not reranker_model or not documents:
        return documents
    
    try:
        # Get embeddings for query and documents
        query_embedding = reranker_model.encode([query])
        doc_embeddings = reranker_model.encode([doc.page_content for doc in documents])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Sort documents by similarity
        doc_sim_pairs = list(zip(documents, similarities))
        doc_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, sim in doc_sim_pairs]
    except:
        return documents

def format_history(msgs):
    """Enhanced chat history formatting"""
    if not msgs: 
        return ""
    
    # Keep more recent context but be more selective
    recent = msgs[-6:] if len(msgs) > 6 else msgs
    hist = []
    
    for i in range(0, len(recent), 2):
        if i+1 < len(recent):
            user_msg = recent[i]['content']
            assistant_msg = recent[i+1]['content']
            
            # Truncate long responses but keep key information
            if len(assistant_msg) > 150:
                assistant_msg = assistant_msg[:150] + "..."
            
            hist.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
    
    return "\n\n".join(hist)

def get_answer(q, chat_history):
    """Enhanced answer generation with better retrieval and context handling"""
    ensure_event_loop()
    
    # Expand and preprocess query
    expanded_query = expand_query(q)
    processed_query = preprocess_question(expanded_query)
    
    # Retrieve documents
    retriever = get_ensemble_retriever()
    docs = retriever.invoke(processed_query)
    
    # Enhanced document scoring and filtering
    reranker = get_reranker_model()
    if reranker:
        docs = rerank_documents(processed_query, docs, reranker)
    
    # Score documents based on multiple factors
    scored_docs = []
    query_terms = set(processed_query.lower().split())
    
    for doc in docs:
        content_lower = doc.page_content.lower()
        score = 0
        
        # Keyword matching score
        content_terms = set(content_lower.split())
        keyword_score = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
        score += keyword_score * 0.4
        
        # FAQ priority boost
        if doc.metadata.get('is_faq'):
            score += 0.3
        
        # Topic relevance boost
        topic = doc.metadata.get('topic', '')
        if any(term in processed_query.lower() for term in [topic]) and topic != 'general':
            score += 0.2
        
        # XID specific terms boost
        xid_terms = ['xid', 'identification', 'creation', 'modification']
        if any(term in content_lower for term in xid_terms):
            score += 0.1
        
        scored_docs.append((doc, score))
    
    # Sort by score and select top documents
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:6]]
    
    # Create focused context
    context_parts = []
    total_length = 0
    max_context_length = 3000  # Limit context length
    
    for doc in top_docs:
        if total_length + len(doc.page_content) <= max_context_length:
            context_parts.append(doc.page_content)
            total_length += len(doc.page_content)
        else:
            # Add truncated version if space allows
            remaining_space = max_context_length - total_length
            if remaining_space > 100:
                context_parts.append(doc.page_content[:remaining_space-3] + "...")
            break
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Generate answer
    llm_chain = get_llm_chain()
    answer = llm_chain.run(
        context=context, 
        question=q, 
        chat_history=chat_history
    )
    
    return answer, top_docs

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

        # Add user message to session
        session.setdefault('messages', []).append({"role": "user", "content": user_message})
        
        # Format chat history
        chat_history = format_history(session['messages'][:-1])
        
        # Get answer with enhanced processing
        answer, _ = get_answer(user_message, chat_history)
        
        # Add assistant response to session
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
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
