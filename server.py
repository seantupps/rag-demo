# file:///D:/Projects/google/frontend/index.html
# python server.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from rag_pipeline import TeslaRAG

app = FastAPI(title="Tesla 10-K RAG API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG
rag = TeslaRAG()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    chartData: dict | None = None

import traceback

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        print(f"--- Incoming Query: {request.query} ---")
        result = rag.query(request.query)
        print(f"--- RAG Result Obtained ---")
        
        # Log specifically if chartData is present
        if result.get("chartData"):
            print(f"Chart Data detected: {list(result['chartData'].keys())}")

        response = QueryResponse(
            answer=result["answer"], 
            sources=result["sources"],
            chartData=result.get("chartData")
        )
        print("--- Response Serialized Successfully ---")
        return response
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
