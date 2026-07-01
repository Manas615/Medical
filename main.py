import os
import sys
import json
import logging
from dotenv import load_dotenv

from fhir_client import FHIRClient
from jsonrpc import JSONRPCDispatcher
from agent import MedicalAgent

# 1. Setup Robust Logging
logger = logging.getLogger("medical_agent")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

# File Handler - captures everything at DEBUG level
file_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
fh = logging.FileHandler("agent_activity.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(file_formatter)
logger.addHandler(fh)

# Console Handler - cleaner output for interactive CLI experience
console_formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(console_formatter)
logger.addHandler(ch)

def run_query(dispatcher: JSONRPCDispatcher, patient_id: str, query: str):
    """Formulates a JSON-RPC request for analyze_history and logs/prints execution results."""
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
    rpc_response = dispatcher.handle_request(rpc_request)
    logger.debug(f"Received JSON-RPC response: {json.dumps(rpc_response)}")
    
    if rpc_response and "result" in rpc_response:
        result = rpc_response["result"]
        # Print the response directly to stdout for user readability
        print("\n=== AGENT RESPONSE (JSON) ===")
        print(json.dumps(result, indent=2))
        print("=======================================")
    elif rpc_response and "error" in rpc_response:
        error = rpc_response["error"]
        logger.error(f"Execution failed with JSON-RPC error:")
        print(json.dumps(error, indent=2))
    else:
        logger.error("Failed to receive a valid response from the JSON-RPC agent pipeline.")

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("CLAUDE_CREDENTIALS") or os.environ.get("ANTHROPIC_API_KEY")
    
    logger.info("======================================================================")
    logger.info("       SMART on FHIR Patient Medical History Analyzer Agent")
    logger.info("======================================================================")
    
    if api_key:
        logger.info("Anthropic API Key detected. Agent will run in LIVE AI mode.")
    else:
        logger.info("No Anthropic API Key found. Agent will run in OFFLINE SIMULATION mode.")
        logger.info("To use live AI, set CLAUDE_CREDENTIALS=your_key in your shell or .env")
    logger.info("Activity details are being recorded to 'agent_activity.log'.")
    logger.info("----------------------------------------------------------------------")

    # Patient setup (default target is smart-1288992)
    patient_id = "smart-1288992"

    # Initialize components
    fhir_client = FHIRClient()
    dispatcher = JSONRPCDispatcher()
    agent = MedicalAgent(fhir_client, dispatcher, api_key)

    # Ingest the patient data once at start to populate cache
    logger.info("Pre-fetching patient history to initialize session cache...")
    try:
        agent.active_bundle = fhir_client.fetch_patient_bundle(patient_id)
        logger.info("Patient history cached successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize patient bundle cache: {e}")
        return

    # Check if user passed arguments for a one-off run
    if len(sys.argv) > 1:
        # Check if first arg is an ID
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

        logger.info(f"Analyzing history for patient ID: {patient_id}")
        logger.info(f"User Query: '{query}'")
        run_query(dispatcher, patient_id, query)
    else:
        # Interactive session Q&A loop (REPL)
       
        try:
            while True:
                print()
                query = input("Ask a question > ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit"):
                    logger.info("Ending interactive session. Goodbye!")
                    break
                
                logger.info(f"Querying: '{query}'...")
                run_query(dispatcher, patient_id, query)
        except KeyboardInterrupt:
            logger.info("\nSession interrupted. Goodbye!")

if __name__ == "__main__":
    main()
