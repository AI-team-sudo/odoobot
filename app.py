from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, BaseSettings
from typing import Optional, List
from pinecone import Pinecone
import os
from sentence_transformers import SentenceTransformer
import requests
from huggingface_hub import InferenceClient
import re
import uvicorn

class Settings(BaseSettings):
    pinecone_key: str
    token: str
    port: int = 10000

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()

# Initialize settings
try:
    settings = get_settings()
except Exception as e:
    raise Exception(f"Failed to load settings: {str(e)}")

# Initialize FastAPI
app = FastAPI(
    title="SQL Query Generator API",
    description="API for generating SQL queries from natural language questions",
    version="1.0.0"
)

# Initialize Pinecone
try:
    pc = Pinecone(api_key=settings.pinecone_key)
    index = pc.Index("odoobot")
except Exception as e:
    raise Exception(f"Failed to initialize Pinecone: {str(e)}")

# Initialize GTE-large model for embeddings
try:
    embedding_model = SentenceTransformer('thenlper/gte-large')
except Exception as e:
    raise Exception(f"Failed to load embedding model: {str(e)}")

# Initialize Hugging Face client
try:
    client = InferenceClient(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        token=settings.token
    )
except Exception as e:
    raise Exception(f"Failed to initialize Hugging Face client: {str(e)}")

class Query(BaseModel):
    question: str

def get_embedding(text: str) -> List[float]:
    """Generate embeddings for input text"""
    try:
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

def get_relevant_schema(question: str) -> str:
    """Retrieve relevant schema from Pinecone"""
    try:
        question_embedding = get_embedding(question)

        results = index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )

        relevant_schema = ""
        for match in results['matches']:
            relevant_schema += match.metadata['schema'] + "\n"

        return relevant_schema
    except Exception as e:
        raise Exception(f"Error retrieving schema: {str(e)}")

def extract_sql_query(text: str) -> str:
    """Extract pure SQL query from the response"""
    try:
        # Try to find SQL query between code blocks
        code_block_match = re.search(r'```(?:sql|vbnet)?\n(.*?)\n```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # If no code blocks, try to find the first statement that looks like SQL
        sql_patterns = [
            r'SELECT.*?;',
            r'INSERT.*?;',
            r'UPDATE.*?;',
            r'DELETE.*?;'
        ]

        for pattern in sql_patterns:
            sql_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if sql_match:
                return sql_match.group(0).strip()

        # If no SQL-like statements found, return the cleaned text
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting SQL query: {str(e)}")

def generate_sql_query(question: str, schema: str) -> str:
    """Generate SQL query using Mixtral model"""
    try:
        prompt = f"""<s>[INST] You are a SQL expert. Given the following database schema:
        {schema}

        Generate a SQL query for the following question:
        {question}

        Important: Return ONLY the raw SQL query. No explanations, no code blocks, no formatting. Just the SQL query that starts with SELECT, INSERT, UPDATE, or DELETE and ends with a semicolon. [/INST]"""

        response = client.text_generation(
            prompt,
            max_new_tokens=1000,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True
        )

        raw_response = response.split("[/INST]")[-1].strip()
        clean_query = extract_sql_query(raw_response)

        return clean_query

    except Exception as e:
        raise Exception(f"Error generating SQL query: {str(e)}")

@app.post("/generate-sql/")
async def generate_sql(query: Query):
    """API endpoint to generate SQL queries"""
    try:
        relevant_schema = get_relevant_schema(query.question)
        sql_query = generate_sql_query(query.question, relevant_schema)
        return Response(content=sql_query, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    print("API initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    print("API shutting down")

if __name__ == "__main__":
    port = int(os.getenv("PORT", settings.port))
    uvicorn.run(app, host="0.0.0.0", port=port)
