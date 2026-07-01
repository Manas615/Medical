import os
import json
import logging
import unittest
from typing import Dict, List, Any

# Import our project modules
from fhir_client import FHIRClient
from jsonrpc import JSONRPCDispatcher, JSONRPCError, METHOD_NOT_FOUND, INVALID_PARAMS
from agent import MedicalAgent, MedicalRecommendation

class TestMedicalAgentProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure logging to write to verify_activity.log to prevent polluting main log
        cls.logger = logging.getLogger("medical_agent")
        cls.logger.setLevel(logging.DEBUG)
        
        # Setup handlers
        if not cls.logger.handlers:
            fh = logging.FileHandler("agent_activity.log", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s'))
            cls.logger.addHandler(fh)

    def test_fhir_client_mock_fallback(self):
        """Test FHIRClient falls back to mock data if live endpoint lacks clinical records."""
        client = FHIRClient()
        # Fetching patient smart-1288992 should succeed and fallback (if empty) or merge
        bundle = client.fetch_patient_bundle("smart-1288992")
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.get("resourceType"), "Bundle")

        # Test demographic extraction
        pat_info = client.extract_patient_info(bundle)
        self.assertEqual(pat_info.get("name"), "Joe Smart")
        self.assertEqual(pat_info.get("gender"), "male")
        self.assertEqual(pat_info.get("birth_date"), "1978-05-15")

        # Test condition extraction
        conditions = client.extract_conditions(bundle)
        self.assertGreater(len(conditions), 0)
        cond_displays = [c["display"] for c in conditions]
        self.assertIn("Type 2 diabetes mellitus", cond_displays)
        self.assertIn("Essential hypertension", cond_displays)

        # Test vitals/observations extraction
        vitals = client.extract_vitals(bundle)
        self.assertGreater(len(vitals), 0)
        vital_displays = [v["display"] for v in vitals]
        self.assertIn("HbA1c", vital_displays)
        self.assertIn("Blood Pressure", vital_displays)

    def test_json_rpc_dispatcher(self):
        """Test JSONRPCDispatcher handles requests, errors, validation, and parameters."""
        dispatcher = JSONRPCDispatcher()

        # Register dummy methods
        def add(a: int, b: int) -> int:
            return a + b
        
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        dispatcher.register("add", add)
        dispatcher.register("greet", greet)

        # 1. Test success
        req_add = {"jsonrpc": "2.0", "method": "add", "params": {"a": 5, "b": 10}, "id": "test_1"}
        res_add = dispatcher.handle_request(req_add)
        self.assertEqual(res_add.get("result"), 15)
        self.assertEqual(res_add.get("id"), "test_1")

        # 2. Test method not found
        req_missing = {"jsonrpc": "2.0", "method": "subtract", "params": {"a": 5, "b": 10}, "id": "test_2"}
        res_missing = dispatcher.handle_request(req_missing)
        self.assertIn("error", res_missing)
        self.assertEqual(res_missing["error"]["code"], METHOD_NOT_FOUND)

        # 3. Test invalid params
        req_bad_params = {"jsonrpc": "2.0", "method": "add", "params": {"x": 5}, "id": "test_3"}
        res_bad_params = dispatcher.handle_request(req_bad_params)
        self.assertIn("error", res_bad_params)
        self.assertEqual(res_bad_params["error"]["code"], INVALID_PARAMS)

    def test_react_agent_offline_flow(self):
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
        res = dispatcher.handle_request(req)
        
        # Verify the structure is correct
        self.assertIn("result", res)
        result = res["result"]
        
        # Pydantic will validate the schema contents
        validated = MedicalRecommendation(**result)
        self.assertIsNotNone(validated.soap)
        self.assertTrue(len(validated.soap.subjective) > 0)
        self.assertTrue(len(validated.soap.objective) > 0)
        self.assertTrue(len(validated.soap.assessment) > 0)
        self.assertGreater(len(validated.soap.plan), 0)
        self.assertGreaterEqual(validated.confidence_score, 0.0)
        self.assertLessEqual(validated.confidence_score, 1.0)

    def test_logging_activity_check(self):
        """Verify that agent_activity.log has been created and has trace contents."""
        log_path = "agent_activity.log"
        self.assertTrue(os.path.exists(log_path))
        
        # Read file contents and verify key log items exist
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # We check for general logs written during test setup and executions
        self.assertTrue(
            "Ingesting patient data" in content or "Mock FHIR Data Loaded" in content,
            "FHIR data ingestion should be logged."
        )
        self.assertTrue(
            "JSON-RPC Request:" in content or "JSON-RPC Response" in content,
            "JSON-RPC payloads should be logged."
        )
        self.assertTrue(
            "[Agent Thought-Process]" in content,
            "Agent thought processes should be logged."
        )

if __name__ == "__main__":
    unittest.main()
