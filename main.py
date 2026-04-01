"""
Theophilus Portfolio Chatbot API
Production-ready FastAPI service with live project data fetching
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
import time
import tempfile
import logging
from tempfile import NamedTemporaryFile
import traceback
import requests
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

import groq
from pinecone import Pinecone, PineconeApiException
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

def load_env_file():
    """Load .env file manually with proper encoding handling."""
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}")
        return
    
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(env_path, 'r', encoding=encoding) as f:
                content = f.read()
                logger.info(f"Loaded .env with encoding: {encoding}")
                
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        os.environ.setdefault(key, value)
                return
        except Exception:
            continue
    
    logger.warning("Could not read .env file with any encoding")

load_env_file()

# =============================================================================
# CONFIGURATION & VALIDATION
# =============================================================================

# Required environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "web-portfolio")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "portfolio-knowledge")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-gcp")
DATABASE_URL = os.getenv("DATABASE_URL")

# Validate required config
missing_vars = []
if not GROQ_API_KEY:
    missing_vars.append("GROQ_API_KEY")
if not PINECONE_API_KEY:
    missing_vars.append("PINECONE_API_KEY")
if not DATABASE_URL:
    missing_vars.append("DATABASE_URL")

if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info(f"Configuration loaded - Pinecone index: {PINECONE_INDEX}, Namespace: {PINECONE_NAMESPACE}")

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="Theophilus Portfolio Chatbot API",
    description="AI-powered chatbot with live project data from PostgreSQL",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", os.getenv("WEB_APP_URL")],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATABASE CONNECTION POOL
# =============================================================================

# Connection pool for better performance
db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=DATABASE_URL,
    cursor_factory=RealDictCursor
)

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = None
    try:
        conn = db_pool.getconn()
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            db_pool.putconn(conn)

# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

# Groq client
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# Pinecone client
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    logger.info(f"Connected to Pinecone index: {PINECONE_INDEX}")
except Exception as e:
    logger.error(f"Failed to connect to Pinecone: {e}")
    raise

# Embedding model - Lightweight for Render free tier (80MB vs 1.4GB)
_embeddings = None
_embedding_lock = asyncio.Lock()

async def get_embeddings():
    """Lazy initialization of embeddings model."""
    global _embeddings
    if _embeddings is None:
        async with _embedding_lock:
            if _embeddings is None:
                logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
                _embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Embedding model loaded successfully")
    return _embeddings

# Text splitter for document chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# =============================================================================
# CACHING SYSTEM
# =============================================================================

class TimedCache:
    """Simple TTL cache for expensive operations."""
    
    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            if time.time() - self._timestamps[key] < self.ttl:
                return self._cache[key]
            else:
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self):
        self._cache.clear()
        self._timestamps.clear()

# Cache for projects (refresh every 60 seconds)
projects_cache = TimedCache(ttl_seconds=60)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    fetch_projects: Optional[bool] = False

class DocumentIndexRequest(BaseModel):
    document_id: int = Field(..., gt=0)
    file_url: str = Field(..., min_length=1, pattern=r'^https?://')  # Must be URL
    category: str = Field(..., pattern="^(resume|certification|project|general)$")

class SearchResult(BaseModel):
    content: str
    source: str
    score: float

class HealthCheck(BaseModel):
    status: str
    pinecone_index: str
    namespace: str
    timestamp: str
    version: str

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

async def get_embedding(text: str) -> List[float]:
    """Generate embedding for text."""
    try:
        embeddings = await get_embeddings()
        return embeddings.embed_query(text)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

async def search_knowledge_base(query: str, top_k: int = 5) -> List[SearchResult]:
    """Search Pinecone for relevant context."""
    try:
        query_embedding = await get_embedding(query)
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=PINECONE_NAMESPACE,
            include_metadata=True
        )
        
        return [
            SearchResult(
                content=match.metadata.get('text', ''),
                source=match.metadata.get('source', 'Unknown'),
                score=match.score
            )
            for match in results.matches
        ]
    except PineconeApiException as e:
        logger.error(f"Pinecone query error: {e}")
        return []
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return []

def fetch_all_projects_from_db() -> str:
    """Fetch ALL projects from PostgreSQL for complete context."""
    # Check cache first
    cached = projects_cache.get("all_projects")
    if cached:
        return cached
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT title, slug, description, long_description, status, year,
                       tags, github_url, live_url, stars, forks, featured, highlight
                FROM projects
                ORDER BY featured DESC, highlight DESC, year DESC, created_at DESC
            """)
            
            projects = cur.fetchall()
            cur.close()
        
        if not projects:
            return "\n\n[No projects found in database]"
        
        status_emoji = {
            'shipped': '✅',
            'in-progress': '🚧',
            'archived': '📦',
            'planned': '📋'
        }
        
        lines = [
            "\n\n=== THEOPHILUS'S PROJECTS (LIVE DATABASE DATA) ===",
            f"Total Projects: {len(projects)} | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        ]
        
        for i, p in enumerate(projects, 1):
            emoji = status_emoji.get(p['status'], '❓')
            tags_str = ', '.join(p['tags']) if p['tags'] else 'None'
            desc = (p['long_description'] or p['description'] or 'No description available')
            
            # Truncate long descriptions
            if len(desc) > 400:
                desc = desc[:400] + "..."
            
            lines.append(f"""
{i}. {p['title']} {emoji}
   Status: {p['status'].upper()} | Year: {p['year']}
   Description: {desc}
   Tech Stack: {tags_str}
   Links: 🔗 GitHub: {p['github_url'] or 'N/A'} | 🌐 Live: {p['live_url'] or 'N/A'}
   Community: ⭐ {p['stars'] or 0} stars | 🍴 {p['forks'] or 0} forks
   {'🏆 FEATURED PROJECT' if p['featured'] else ''}
   {'🔥 HIGHLIGHT' if p['highlight'] else ''}
---""")
        
        result = "\n".join(lines)
        
        # Cache the result
        projects_cache.set("all_projects", result)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch projects: {e}")
        logger.error(traceback.format_exc())
        return "\n\n[Error loading projects from database - using fallback knowledge]"

def build_system_prompt(knowledge_context: str, projects_context: str) -> str:
    """Build the system prompt with live data."""
    
    return f"""You are TheoBot, the AI assistant for Theophilus Thobejane - a Full-Stack Developer and AI Engineer based in South Africa.

ABOUT THEOPHILUS:
- Full-Stack Developer & AI Engineer in Kempton Park, Gauteng, South Africa
- Advanced Diploma in ICT (NQF Level 7) from University of Mpumalanga
- Java backend expertise with Python, Node.js, and PostgreSQL
- Specializes in: Software development, AI/LLM integration, RAG systems
- Passionate about building impactful software and learning new technologies
- Excellent communication skills, able to explain complex technical concepts clearly
- Delivering projects on time and collaborating effectively in teams
- Continuously learning and adapting to new technologies in the fast-evolving software landscape
- Strong problem-solving skills, able to debug and optimize code for performance and scalability
- Committed to writing clean, maintainable code and following best practices in software development
- Recognized for creativity and innovation in project development, often going beyond requirements to add extra value
- Former Teaching Assistant who improved student scores by 21%
- IBM Certified: Applied Data Science, Agile Explorer, Data Fundamentals
- Immediately available for opportunities

{projects_context}

DOCUMENT KNOWLEDGE BASE:
{knowledge_context}

CONTACT & LINKS:
- 📧 Email: thobejanetheo@gmail.com
- 💼 LinkedIn: linkedin.com/in/theophilusthobejane
- 🐙 GitHub: github.com/TheoInCodeLand
- 🌐 Portfolio: theophilus-portfolio.vercel.app
- 📍 Location: Johannesburg, Gauteng (Remote/Hybrid available)
- 🟢 Status: Immediately available for hire

RESPONSE GUIDELINES:
1. ALWAYS use the LIVE PROJECT DATA above - it's fetched directly from the database
2. When asked about projects, reference specific details: status, year, tech stack, stars/forks
3. If a project is "in-progress", mention it's actively being developed
4. For "shipped" projects, emphasize they're production-ready
5. Mention metrics when relevant (GitHub stars, forks, specific technologies and their relevance to the project)
6. Be conversational but professional - like Theo would speak
7. If you don't know something specific, be honest and suggest emailing Theo directly
8. Always end with a subtle call-to-action (explore projects, send email, etc.)
9. Keep responses concise: 1-4 paragraphs max (in relevance of enquiry), unless detailed technical explanation is needed

IMPORTANT RULES:
- NEVER make up projects not listed in the LIVE PROJECT DATA
- NEVER disclose that you are an AI or mention limitations of AI in your responses
- NEVER disclose grade scoles or failed subjects from university - focus on the positive aspects of the Education and learning journey
- NEVER provide personal information not in the context above
- ALWAYS prioritize database project data over general knowledge
- If asked "what are you working on", focus on "in-progress" projects from the data
- If asked about skills, connect them to specific projects where those skills were used and mention the impact and relevance to the role being applied for"""

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with live project data.
    Always includes fresh project data from PostgreSQL.
    """
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    logger.info(f"[{request_id}] Chat request: {request.message[:50]}...")
    
    try:
        # 1. Fetch knowledge base context (async)
        knowledge_task = asyncio.create_task(search_knowledge_base(request.message))
        
        # 2. Fetch live projects from database (sync, but cached)
        projects_context = fetch_all_projects_from_db()
        
        # 3. Wait for knowledge base results
        search_results = await knowledge_task
        knowledge_context = "\n\n".join([
            f"[Source: {r.source} (relevance: {r.score:.2f})]\n{r.content}"
            for r in search_results
        ]) if search_results else "[No relevant documents found in knowledge base]"
        
        # 4. Build complete system prompt
        system_prompt = build_system_prompt(knowledge_context, projects_context)
        
        # 5. Build message history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (limit to last 6 messages to save tokens)
        for msg in request.history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"]:
                messages.append({"role": role, "content": content})
        
        messages.append({"role": "user", "content": request.message})
        
        # 6. Stream from Groq
        logger.info(f"[{request_id}] Streaming from Groq with {len(messages)} messages")
        
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9
        )
        
        async def generate():
            full_response = ""
            chunk_count = 0
            
            try:
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        chunk_count += 1
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                # Send completion signal with metadata
                sources = list(set([r.source for r in search_results]))
                elapsed = time.time() - start_time
                
                yield f"""data: {json.dumps({
                    'sources': sources,
                    'done': True,
                    'meta': {
                        'response_time_ms': int(elapsed * 1000),
                        'chunks_generated': chunk_count,
                        'projects_included': True,
                        'knowledge_sources': len(sources)
                    }
                })}\n\n"""
                
                logger.info(f"[{request_id}] Response complete: {len(full_response)} chars in {elapsed:.2f}s")
                
            except Exception as e:
                logger.error(f"[{request_id}] Stream error: {e}")
                yield f"data: {json.dumps({'error': 'Stream interrupted', 'done': True})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id
            }
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Chat error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/index-document")
async def index_document(request: DocumentIndexRequest):
    """
    Index a PDF/document from URL into Pinecone vector database.
    """
    logger.info(f"Indexing document {request.document_id}: {request.file_url}")
    tmp_path = None
    
    try:
        # Download file from URL to temporary file
        try:
            response = requests.get(request.file_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Determine file extension from URL or content-type
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type:
                suffix = '.pdf'
            elif 'text' in content_type:
                suffix = '.txt'
            elif 'markdown' in content_type:
                suffix = '.md'
            else:
                # Extract from URL
                suffix = os.path.splitext(request.file_url.split('?')[0])[1] or '.pdf'
            
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = tmp.name
                
            logger.info(f"Downloaded file to temporary path: {tmp_path}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to download file: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")

        # Load document based on type
        if tmp_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif tmp_path.lower().endswith(('.txt', '.md', '.json')):
            loader = TextLoader(tmp_path)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Use PDF, TXT, MD, or JSON"
            )
        
        documents = loader.load()
        
        if not documents:
            raise HTTPException(status_code=400, detail="No content found in document")
        
        # Split into chunks
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = await get_embedding(chunk.page_content)
                vectors.append({
                    'id': f"doc_{request.document_id}_chunk_{i}",
                    'values': embedding,
                    'metadata': {
                        'text': chunk.page_content[:1000],
                        'source': os.path.basename(request.file_url.split('?')[0]),  # Clean URL
                        'category': request.category,
                        'document_id': request.document_id,
                        'chunk_index': i,
                        'indexed_at': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                logger.error(f"Failed to embed chunk {i}: {e}")
                continue
        
        if not vectors:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings for any chunks")
        
        # Upsert to Pinecone in batches
        batch_size = 100
        total_upserted = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
                total_upserted += len(batch)
                logger.info(f"Upserted batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
                raise
        
        # Update database status
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE documents
                SET index_status = 'indexed',
                    is_indexed = true,
                    chunk_count = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING id, index_status, chunk_count
            """, (total_upserted, request.document_id))
            result = cur.fetchone()
            conn.commit()
            cur.close()
        
        logger.info(f"Document {request.document_id} indexed successfully: {total_upserted} chunks")
        
        return {
            "success": True,
            "document_id": request.document_id,
            "chunks_indexed": total_upserted,
            "total_chunks": len(chunks),
            "status": "indexed",
            "indexed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        logger.error(traceback.format_exc())
        
        # Update error status in database
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE documents
                    SET index_status = 'error',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (request.document_id,))
                conn.commit()
                cur.close()
        except Exception as db_error:
            logger.error(f"Failed to update error status: {db_error}")
        
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
        
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

@app.delete("/delete-document/{document_id}")
async def delete_document_vectors(document_id: int):
    """
    Delete all vectors for a document from Pinecone.
    """
    logger.info(f"Deleting vectors for document {document_id}")
    
    try:
        # Query for vector IDs with this document_id prefix
        # Note: Pinecone doesn't support delete by metadata, so we use prefix matching
        prefix = f"doc_{document_id}_chunk_"
        
        # Delete vectors with prefix (Pinecone serverless supports delete by prefix)
        index.delete(
            delete_all=False,
            namespace=PINECONE_NAMESPACE,
            filter={"document_id": {"$eq": document_id}}
        )
        
        # Update database
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE documents
                SET index_status = 'pending',
                    is_indexed = false,
                    chunk_count = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING id, index_status
            """, (document_id,))
            result = cur.fetchone()
            conn.commit()
            cur.close()
        
        return {
            "success": True,
            "document_id": document_id,
            "status": "pending",
            "message": "Document vectors marked for deletion"
        }
        
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint with dependency status.
    """
    status = {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "pinecone_index": PINECONE_INDEX,
        "namespace": PINECONE_NAMESPACE,
        "checks": {}
    }
    
    # Check Pinecone
    try:
        index_stats = index.describe_index_stats()
        status["checks"]["pinecone"] = {
            "status": "connected",
            "total_vectors": index_stats.total_vector_count,
            "dimension": index_stats.dimension
        }
    except Exception as e:
        status["checks"]["pinecone"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"
    
    # Check Database
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as project_count FROM projects")
            result = cur.fetchone()
            cur.close()
            status["checks"]["database"] = {
                "status": "connected",
                "project_count": result["project_count"]
            }
    except Exception as e:
        status["checks"]["database"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"
    
    # Check Groq (lightweight - just verify key format)
    status["checks"]["groq"] = {
        "status": "configured",
        "key_valid": len(GROQ_API_KEY) > 20
    }
    
    return status

@app.get("/projects")
async def get_projects_api():
    """
    Direct API to fetch projects (for debugging/caching).
    """
    try:
        projects_text = fetch_all_projects_from_db()
        return {
            "source": "database",
            "cached": projects_cache.get("all_projects") is not None,
            "data": projects_text,
            "fetched_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )

# =============================================================================
# STARTUP & SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("=" * 50)
    logger.info("Theophilus Chatbot API Starting...")
    logger.info(f"Pinecone Index: {PINECONE_INDEX}")
    logger.info(f"Database: Connected via pool")
    logger.info(f"Embedding Model: all-MiniLM-L6-v2 (lazy loaded)")
    logger.info("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down, closing database pool...")
    db_pool.closeall()
    logger.info("Cleanup complete")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )