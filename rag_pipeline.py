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
                "description": "REQUIRED TOOL for visualizing quantitative financial data. Use this whenever the context contains numerical data (revenue, profit, deliveries, etc.) across multiple time periods or categories that benefit from a bar or line chart.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Descriptive title for the chart"},
                        "type": {"type": "string", "enum": ["bar", "line"], "description": "Use 'line' for trends over time, 'bar' for comparisons between categories or years."},
                        "labels": {"type": "array", "items": {"type": "string"}, "description": "X-axis labels (e.g., years '2023', '2024')"},
                        "datasets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "description": "The metric name (e.g., 'Total Revenue ($M)')"},
                                    "data": {"type": "array", "items": {"type": "number"}, "description": "The numeric values extracted from the context"},
                                    "borderColor": {"type": "string", "description": "CSS color for the line/border (default: #E82127 for Tesla Red)"},
                                    "backgroundColor": {"type": "string", "description": "CSS color for the bars/fill (default: rgba(232, 33, 39, 0.2))"}
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
            "about Tesla's 2025 10-K filing with extreme precision. \n\n"
            "VISUALIZATION MANDATE:\n"
            "If the KNOWLEDGE CONTEXT contains numerical data, financial metrics, or performance statistics "
            "that span multiple years or categories, you MUST call the 'render_financial_chart' tool. \n"
            "- Prioritize a 'line' chart ONLY when the y axis has a small range of values.\n"
            "- Always prioritize a 'bar' chart for comparing different business segments or specific annual targets.\n"
            "- If multiple metrics are available (e.g., Revenue and Net Income), include them as separate datasets in the same chart.\n\n"
            "Do not include Totals in the chart unless specifically asked for.\n\n"
            "RULES:\n"
            "1. Use ONLY the provided KNOWLEDGE CONTEXT.\n"
            "2. If the context has numbers but no explicit chart request, CREATE THE CHART ANYWAY to support your analysis.\n"
            "3. STYLE: Professional, analytical, and data-heavy.\n"
            "4. CITATIONS: Cite the specific SEC Item (e.g., [ITEM 1.], [ITEM 1A.], [ITEM 8] Only include the number.) for every fact or figure.\n\n"
            "EXAMPLE 1 (Trend Analysis):\n"
            "User: Compare total revenue for 2023 and 2024.\n"
            "Context: Total revenue was $96,773 million in 2023 and $98,826 million in 2024.\n"
            "Analyst Action: Call `render_financial_chart` with title 'Tesla Total Revenue', type 'line', labels ['2023', '2024'], and data [96773, 98826].\n\n"
            "EXAMPLE 2 (Comparison Snapshot):\n"
            "User: How did vehicle deliveries compare across models in 2024?\n"
            "Context: Model 3/Y deliveries were 1,739,707 while Model S/X and Other models were 68,542.\n"
            "Analyst Action: Call `render_financial_chart` with title '2024 Deliveries by Model', type 'bar', labels ['Model 3/Y', 'Model S/X/Other'], and data [1739707, 68542]."
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
