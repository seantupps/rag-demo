# DO NOT REMOVE
# python rag_pipeline.py

import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Add scripts directory to path for imports
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
from vector_store import VectorStore

load_dotenv()

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
MODEL_ID = "gemini-2.5-flash"

class TeslaRAG:
    def __init__(self):
        # Initialize the VectorStore
        print("Initializing Tesla 10-K Vector Store...")
        self.vector_store = VectorStore(storage_path="data/tesla/vector_store.json")
        if not self.vector_store.load():
            print("Warning: Could not load vector store. Ensure embeddings have been generated.")
        
        # Initialize Vertex AI Client
        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

        # Define tools
        self.tools = [
            {
                "name": "render_financial_chart",
                "description": "Renders a bar or line chart comparing financial metrics (e.g., revenue vs profit) over multiple years.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "The title of the chart"},
                        "type": {"type": "string", "enum": ["bar", "line"], "description": "Type of chart"},
                        "labels": {"type": "array", "items": {"type": "string"}, "description": "X-axis labels (e.g., ['2023', '2024', '2025'])"},
                        "datasets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "description": "The metric name (e.g., 'Total Revenue')"},
                                    "data": {"type": "array", "items": {"type": "number"}, "description": "The numeric values"},
                                    "borderColor": {"type": "string", "description": "CSS color for the line/border"},
                                    "backgroundColor": {"type": "string", "description": "CSS color for the bars/fill"}
                                }
                            }
                        }
                    },
                    "required": ["title", "type", "labels", "datasets"]
                }
            }
        ]

    def query(self, user_query, top_k=3):
        print(f"\nProcessing Query: {user_query}")
        
        # 1. RETRIEVAL
        print(f"Retrieving top {top_k} relevant segments from 10-K...")
        results = self.vector_store.search(user_query, top_k=top_k)
        
        if not results:
            return {"answer": "Error: No relevant data found.", "sources": []}
            
        context_blocks = []
        for r in results:
            source = r['chunk']['source']
            content = r['chunk']['content']
            context_blocks.append(f"--- SOURCE: {source} ---\n{content}\n")
        
        knowledge_context = "\n".join(context_blocks)
        
        # 2. GENERATION
        system_instruction = (
            "You are a Senior Tesla Equity Research Analyst. Your task is to answer questions "
            "about Tesla's 2025 10-K filing with extreme precision and detail. \n\n"
            "TOOLS:\n"
            "Use the 'render_financial_chart' tool when asked to compare financial metrics over time.\n\n"
            "RULES:\n"
            "1. Use ONLY the provided KNOWLEDGE CONTEXT to answer.\n"
            "2. If the answer is not contained in the context, state clearly that the information is unavailable.\n"
            "3. STYLE: Professional, concise, and data-driven.\n"
            "4. CITATIONS: You MUST cite the specific SEC Item (Examples: [ITEM 1.], [ITEM 1A.], [ITEM 8]) for every major fact or number.\n"
            "Place citations at the end of sentences or paragraphs."
        )
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"KNOWLEDGE CONTEXT: \n{knowledge_context}"),
                    types.Part.from_text(text=f"USER QUESTION: {user_query}")
                ]
            )
        ]
        
        try:
            # Use Tool class for explicit tool definition
            tool_list = [types.Tool(function_declarations=self.tools)]
            
            response = self.client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    tools=tool_list,
                ),
            )
            
            chart_data = None
            answer_text = ""
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        fc = part.function_call
                        if fc.name == "render_financial_chart":
                            # Ensure args is serializable as a dict
                            chart_data = dict(fc.args) if fc.args else None
                    if part.text:
                        answer_text += part.text
            
            # Fallback if no text but tool was called
            if not answer_text.strip() and chart_data:
                answer_text = "Analysis complete. I've generated a chart comparing the requested financial metrics."
            elif not answer_text.strip():
                answer_text = "I couldn't find specific details for that in the 10-K. Could you try rephrasing?"

            return {
                "answer": answer_text,
                "sources": [r['chunk'] for r in results],
                "chartData": chart_data
            }
        except Exception as e:
            print(f"ERROR in TeslaRAG.query: {e}")
            return {
                "answer": f"The analyst encountered an error processing your request: {str(e)}",
                "sources": [],
                "chartData": None
            }

if __name__ == "__main__":
    try:
        rag = TeslaRAG()
        
        test_queries = [
            "What's Tesla's biggest risk factor in 2025?"
        ]
        
        for q in test_queries:
            print("\n" + "="*80)
            print(f"TESTING ANALYST QUERY: {q}")
            print("-" * 80)
            result = rag.query(q)
            print(f"RESPONSE: {result['answer']}")
            print("="*80)
    except Exception as e:
        print(f"FATAL ERROR during pipeline execution: {e}")
