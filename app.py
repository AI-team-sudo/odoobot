from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
import os
import re
import uvicorn
from functools import lru_cache

class Settings(BaseSettings):
    pinecone_key: str
    token: str
    port: int = 10000

    model_config = SettingsConfigDict(env_file=".env")

@lru_cache()
def get_settings():
    return Settings()

app = FastAPI(
    title="SQL Query Generator API",
    description="API for generating SQL queries from natural language questions",
    version="1.0.0"
)

# Lazy loading of models and clients
embedding_model = None
pinecone_client = None
hf_client = None

def initialize_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('thenlper/gte-base')  # Using smaller model
    return embedding_model

def initialize_pinecone():
    global pinecone_client
    if pinecone_client is None:
        from pinecone import Pinecone
        settings = get_settings()
        pinecone_client = Pinecone(api_key=settings.pinecone_key)
    return pinecone_client

def initialize_hf_client():
    global hf_client
    if hf_client is None:
        from huggingface_hub import InferenceClient
        settings = get_settings()
        hf_client = InferenceClient(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            token=settings.token
        )
    return hf_client

class Query(BaseModel):
    question: str

def get_embedding(text: str) -> List[float]:
    """Generate embeddings for input text"""
    try:
        model = initialize_embedding_model()
        embedding = model.encode(text, convert_to_tensor=False)  # Prevent GPU memory usage
        return embedding.tolist()
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

def get_relevant_schema(question: str) -> str:
    """Retrieve relevant schema from Pinecone"""
    try:
        pc = initialize_pinecone()
        index = pc.Index("odoobot")

        question_embedding = get_embedding(question)

        results = index.query(
            vector=question_embedding,
            top_k=3,  # Reduced from 5 to 3
            include_metadata=True
        )

        relevant_schema = "\n".join(
            match.metadata['schema'] for match in results['matches']
        )

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
        sql_match = re.search(r'(SELECT|INSERT|UPDATE|DELETE).*?;', text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(0).strip()

        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting SQL query: {str(e)}")

def generate_sql_query(question: str, schema: str) -> str:
    """Generate SQL query using Mixtral model"""
    try:
        client = initialize_hf_client()

        prompt = f"""<s>[INST] You are a SQL expert. Given the following database schema:
        {schema}

        Generate a SQL query for the following question:
        {question}

        Important: Return ONLY the raw SQL query. No explanations, no code blocks, no formatting. Just the SQL query that starts with SELECT, INSERT, UPDATE, or DELETE and ends with a semicolon. [/INST]"""

        response = client.text_generation(
            prompt,
            max_new_tokens=500,  # Reduced from 1000
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True
        )

        raw_response = response.split("[/INST]")[-1].strip()
        return extract_sql_query(raw_response)

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
