import os
import sys
import json
import logging
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any

from fhir_client import FHIRClient
from jsonrpc import JSONRPCDispatcher
from agent import MedicalAgent

# 1. Setup Robust Logging to file only (at DEBUG level)
logger = logging.getLogger("medical_agent")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

file_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
fh = logging.FileHandler("agent_activity.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(file_formatter)
logger.addHandler(fh)

def print_human_readable_recommendation(result: Dict[str, Any]):
    """Formats the SOAP note result in a clean, human-readable structure."""
    if "soap" in result:
        soap = result["soap"]
        print("\n" + "=" * 80)
        print("                 CLINICAL ANALYSIS REPORT (SOAP NOTE)")
        print("=" * 80)
        print(f"\n[SUBJECTIVE]\n{soap.get('subjective', 'N/A')}\n")
        print(f"[OBJECTIVE]\n{soap.get('objective', 'N/A')}\n")
        print(f"[ASSESSMENT]\n{soap.get('assessment', 'N/A')}\n")
        print("[PLAN]")
        plan = soap.get("plan", [])
        if isinstance(plan, list):
            for idx, item in enumerate(plan, 1):
                print(f"  {idx}. {item}")
        else:
            print(f"  {plan}")
        print("\n" + "-" * 80)
        print(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
        print("=" * 80 + "\n")
    else:
        # Fallback to json dump if the structure is unexpected
        print(json.dumps(result, indent=2))

async def run_query(dispatcher: JSONRPCDispatcher, patient_id: str, query: str):
    """Sends query to JSON-RPC dispatcher and prints clean output."""
    rpc_request = {
        "jsonrpc": "2.0",
        "method": "analyze_history",
        "params": {
            "patient_id": patient_id,
            "query": query
        },
        "id": "cli_query_001"
    }

    logger.debug(f"Sending JSON-RPC request: {json.dumps(rpc_request)}")
    print("Thinking...")
    rpc_response = await dispatcher.handle_request(rpc_request)
    logger.debug(f"Received JSON-RPC response: {json.dumps(rpc_response)}")
    
    if rpc_response and "result" in rpc_response:
        result = rpc_response["result"]
        print_human_readable_recommendation(result)
    elif rpc_response and "error" in rpc_response:
        print("[ERROR] Analysis failed:")
        print(json.dumps(rpc_response["error"], indent=2))
    else:
        print("[ERROR] Unresponsive pipeline.")

async def main():
    load_dotenv()
    api_key = os.environ.get("CLAUDE_CREDENTIALS") or os.environ.get("ANTHROPIC_API_KEY")
    
    print("======================================================================")
    print("       SMART on FHIR Patient Medical History Analyzer Agent")
    print("======================================================================")
    if api_key:
        print("Live AI Mode Active (Anthropic Claude)")
    else:
        print("Simulation Mode Active (Local reasoning)")
    print("Activity log: agent_activity.log")
    print("----------------------------------------------------------------------")

    patient_id = "smart-1288992"

    fhir_client = FHIRClient()
    dispatcher = JSONRPCDispatcher()
    agent = MedicalAgent(fhir_client, dispatcher, api_key)

    print("Caching patient history...")
    try:
        agent.active_bundle = await fhir_client.fetch_patient_bundle(patient_id)
        print("Cache initialized.")
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        return

    # CLI one-off run
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        query = ""
        if arg1.startswith("smart-") or arg1.isdigit():
            patient_id = arg1
            if len(sys.argv) > 2:
                query = " ".join(sys.argv[2:])
        else:
            query = " ".join(sys.argv[1:])

        if not query:
            query = "Analyze patient's chronic conditions, current control, and suggest a clinical recommendation."

        print(f"\nQuery: '{query}'")
        await run_query(dispatcher, patient_id, query)
    else:
        # Interactive REPL mode
        print("\nEnter questions about the patient's medical history.")
        print("Type 'exit' or 'quit' to close the session.")
        print("----------------------------------------------------------------------")
        try:
            while True:
                # Run standard input query in an executor to avoid blocking the async event loop
                query = await asyncio.to_thread(input, "\nAsk a question > ")
                query = query.strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break
                await run_query(dispatcher, patient_id, query)
        except KeyboardInterrupt:
            print("\nSession ended. Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())

