# DO NOT REMOVE COMMENTS
# 
# python scripts/vector_store.py

import os
import json
import numpy as np
import time
from google import genai
from google.genai import types
from google.api_core import exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorStore:
    def __init__(self, storage_path="data/tesla/vector_store.json", model="gemini-embedding-001"):
        self.storage_path = storage_path
        self.model = model
        # Initialize client for Vertex AI using ADC
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        self.client = genai.Client(
            vertexai=True, 
            project=self.project_id, 
            location=self.location
        )
        self.chunks = []
        self.embeddings = []
        
    def load_chunks(self, chunks_path="data/tesla/chunks.json"):
        if not os.path.exists(chunks_path):
            print(f"Error: {chunks_path} not found.")
            return False
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        return True

    def generate_embeddings(self, model=None):
        model = model or self.model
        
        # Check if we already have progress
        start_index = len(self.embeddings)
        if start_index >= len(self.chunks):
            print("All chunks already have embeddings. Skipping generation.")
            return True
            
        if start_index > 0:
            print(f"Resuming embedding generation from chunk {start_index}/{len(self.chunks)}...")
        else:
            print(f"Generating embeddings for {len(self.chunks)} chunks using {model}...")
        
        # Process in batches to avoid rate limits and API overhead
        batch_size = 20
        
        for i in range(start_index, len(self.chunks), batch_size):
            batch = self.chunks[i:i+batch_size]
            texts = [c['content'] for c in batch]
            
            try:
                response = self.client.models.embed_content(
                    model=model,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                
                # The response structure for batch embedding in google-genai
                batch_embeddings = [e.values for e in response.embeddings]
                self.embeddings.extend(batch_embeddings)
                print(f"Processed {len(self.embeddings)}/{len(self.chunks)}...")
                
                # INCREMENTAL SAVE: Save after every batch so progress isn't lost
                self.save()
                
            except Exception as e:
                print(f"Error generating embeddings for batch starting at {i}: {e}")
                print("Progress has been saved to the vector store.")
                return False
                
        return True

    def save(self):
        print(f"Saving vector store to {self.storage_path}...")
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "model": self.model
        }
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print("Save successful.")

    def load(self):
        if not os.path.exists(self.storage_path):
            print(f"Error: {self.storage_path} not found.")
            return False
        print(f"Loading vector store from {self.storage_path}...")
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data["chunks"]
            self.embeddings = data["embeddings"]
            self.model = data.get("model", self.model)
        return True

    def search(self, query, top_k=5, model=None):
        model = model or self.model
        # Generate embedding for the query
        response = self.client.models.embed_content(
            model=model,
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_embedding = response.embeddings[0].values
        
        # Calculate cosine similarity
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append(sim)
            
        # Get top K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx])
            })
            
        return results

if __name__ == "__main__":
    # Test script
    store = VectorStore(model="gemini-embedding-001") 
    
    # Try to load existing progress first
    if os.path.exists(store.storage_path):
        store.load()
    
    if store.load_chunks():
        if store.generate_embeddings():
            # Final search test (only runs if all embeddings are complete or were already complete)
            print("Vector store is up to date.")
            
            # Simple test search
            test_query = "What was the most popular Tesla Model in 2025?"
            print(f"\nTesting search for: '{test_query}'")
            results = store.search(test_query, top_k=1)
            for r in results:
                print(f"Score: {r['score']:.4f} | Source: {r['chunk']['source']}")
                print(f"Content: {r['chunk']['content']}")
                print("-" * 40)
