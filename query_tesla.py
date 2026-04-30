# python query_tesla.py

import sys
import os
import argparse
from rag_pipeline import TeslaRAG

def main():
    parser = argparse.ArgumentParser(description="Query the Tesla 10-K RAG pipeline via CLI.")
    parser.add_argument("query", nargs="?", help="The question to ask about Tesla's 10-K.")
    args = parser.parse_args()

    # If no query provided as arg, enter interactive mode
    if not args.query:
        print("Tesla Analyst CLI - Type 'exit' to quit.")
        rag = TeslaRAG()
        while True:
            try:
                user_msg = input("\nQuery > ")
                if user_msg.lower() in ["exit", "quit"]:
                    break
                if not user_msg.strip():
                    continue
                
                result = rag.query(user_msg)
                print("\n" + "="*50)
                print(f"ANALYST RESPONSE:\n{result['answer']}")
                
                if result.get('chartData'):
                    print("\n[CHART DATA GENERATED]")
                    print(result['chartData'])
                
                print("\nSOURCES:")
                for i, src in enumerate(result['sources']):
                    print(f"- [{i+1}] {src['source']}")
                print("="*50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Run single query
        rag = TeslaRAG()
        result = rag.query(args.query)
        print(f"\nANALYST RESPONSE: {result['answer']}")
        if result.get('chartData'):
            print(f"\nCHART DATA: {result['chartData']}")

if __name__ == "__main__":
    main()
