import os
import json
import logging
import pytest
from typing import Dict, List, Any

# Import our project modules
from fhir_client import FHIRClient
from jsonrpc import JSONRPCDispatcher, JSONRPCError
from agent import MedicalAgent, MedicalRecommendation

# Configure logging for tests
logger = logging.getLogger("medical_agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler("agent_activity.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s'))
    logger.addHandler(fh)

@pytest.mark.asyncio
async def test_fhir_client_mock_fallback():
    """Test FHIRClient falls back to mock data if live endpoint lacks clinical records."""
    client = FHIRClient()
    # Fetching patient smart-1288992 should succeed and fallback (if empty) or merge
    bundle = await client.fetch_patient_bundle("smart-1288992")
    assert bundle is not None
    assert bundle.get("resourceType") == "Bundle"

    # Test demographic extraction
    pat_info = client.extract_patient_info(bundle)
    assert pat_info.get("name") == "Joe Smart"
    assert pat_info.get("gender") == "male"
    assert pat_info.get("birth_date") == "1978-05-15"

    # Test condition extraction
    conditions = client.extract_conditions(bundle)
    assert len(conditions) > 0
    cond_displays = [c["display"] for c in conditions]
    assert "Type 2 diabetes mellitus" in cond_displays
    assert "Essential hypertension" in cond_displays

    # Test vitals/observations extraction
    vitals = client.extract_vitals(bundle)
    assert len(vitals) > 0
    vital_displays = [v["display"] for v in vitals]
    assert "HbA1c" in vital_displays
    assert "Blood Pressure" in vital_displays

@pytest.mark.asyncio
async def test_json_rpc_dispatcher():
    """Test JSONRPCDispatcher handles requests, errors, validation, and parameters."""
    dispatcher = JSONRPCDispatcher()

    # Register dummy methods (mix of sync and async)
    def add(a: int, b: int) -> int:
        return a + b
    
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    dispatcher.register("add", add)
    dispatcher.register("greet", greet)

    # 1. Test success (sync method)
    req_add = {"jsonrpc": "2.0", "method": "add", "params": {"a": 5, "b": 10}, "id": "test_1"}
    res_add = await dispatcher.handle_request(req_add)
    assert res_add.get("result") == 15
    assert res_add.get("id") == "test_1"

    # 2. Test success (async method)
    req_greet = {"jsonrpc": "2.0", "method": "greet", "params": {"name": "Alice"}, "id": "test_greet"}
    res_greet = await dispatcher.handle_request(req_greet)
    assert res_greet.get("result") == "Hello, Alice!"
    assert res_greet.get("id") == "test_greet"

    # 3. Test method not found
    req_missing = {"jsonrpc": "2.0", "method": "subtract", "params": {"a": 5, "b": 10}, "id": "test_2"}
    res_missing = await dispatcher.handle_request(req_missing)
    assert "error" in res_missing
    assert res_missing["error"]["code"] == -32601

    # 4. Test invalid params
    req_bad_params = {"jsonrpc": "2.0", "method": "add", "params": {"x": 5}, "id": "test_3"}
    res_bad_params = await dispatcher.handle_request(req_bad_params)
    assert "error" in res_bad_params
    assert res_bad_params["error"]["code"] == -32602

@pytest.mark.asyncio
async def test_react_agent_offline_flow():
    """Test the MedicalAgent in offline simulation mode returns validated SOAP recommendation."""
    client = FHIRClient()
    dispatcher = JSONRPCDispatcher()
    agent = MedicalAgent(client, dispatcher, api_key=None)

    # Execute analyze_history via JSON-RPC dispatcher
    query_str = "Check patient diabetes history"
    req = {
        "jsonrpc": "2.0",
        "method": "analyze_history",
        "params": {"patient_id": "smart-1288992", "query": query_str},
        "id": "react_test"
    }
    res = await dispatcher.handle_request(req)
    
    # Verify the structure is correct
    assert "result" in res
    result = res["result"]
    
    # Pydantic will validate the schema contents
    validated = MedicalRecommendation(**result)
    assert validated.soap is not None
    assert len(validated.soap.subjective) > 0
    assert len(validated.soap.objective) > 0
    assert len(validated.soap.assessment) > 0
    assert len(validated.soap.plan) > 0
    assert 0.0 <= validated.confidence_score <= 1.0

@pytest.mark.asyncio
async def test_react_agent_2step_general_recommendation_flow():
    """Test the MedicalAgent in offline simulation mode for a general recommendation query runs the 2-step process."""
    client = FHIRClient()
    dispatcher = JSONRPCDispatcher()
    agent = MedicalAgent(client, dispatcher, api_key=None)

    # General recommendation query
    query_str = "Analyze patient's chronic conditions, current control, and suggest a clinical recommendation."
    req = {
        "jsonrpc": "2.0",
        "method": "analyze_history",
        "params": {"patient_id": "smart-1288992", "query": query_str},
        "id": "react_2step_test"
    }
    res = await dispatcher.handle_request(req)
    
    assert "result" in res
    result = res["result"]
    
    # Validate result
    validated = MedicalRecommendation(**result)
    assert validated.soap is not None
    assert len(validated.soap.subjective) > 0
    assert len(validated.soap.objective) > 0
    assert len(validated.soap.assessment) > 0
    assert len(validated.soap.plan) > 0
    assert 0.0 <= validated.confidence_score <= 1.0

def test_logging_activity_check():
    """Verify that agent_activity.log has been created and has trace contents."""
    log_path = "agent_activity.log"
    assert os.path.exists(log_path)
    
    # Read file contents and verify key log items exist
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "Ingesting patient" in content or "Mock FHIR Data Loaded" in content
    assert "JSON-RPC Request:" in content or "JSON-RPC Response" in content
    assert "[Agent Thought-Process]" in content
