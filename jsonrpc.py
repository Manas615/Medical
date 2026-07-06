import json
import logging
from typing import Dict, Any, Callable, Tuple, Optional

logger = logging.getLogger("medical_agent")

# Standard JSON-RPC 2.0 Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        err = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err

class JSONRPCDispatcher:
    def __init__(self):
        self.methods: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        """Registers a function to be callable via JSON-RPC."""
        self.methods[name] = func
        logger.debug(f"Registered JSON-RPC method: {name}")

    async def handle_request_string(self, request_str: str) -> str:
        """Parses a JSON string request, processes it, and returns the response as a JSON string."""
        logger.debug(f"Received raw JSON-RPC string: {request_str}")
        try:
            request_data = json.loads(request_str)
        except json.JSONDecodeError as e:
            response = self._make_error_response(PARSE_ERROR, "Parse error: Invalid JSON.", None, None)
            return json.dumps(response)

        response_data = await self.handle_request(request_data)
        
        # If it is a list of requests (batch request)
        if isinstance(response_data, list):
            # Omit notifications that have no response
            valid_responses = [r for r in response_data if r is not None]
            if not valid_responses:
                return ""
            return json.dumps(valid_responses)
        
        if response_data is None:
            return ""
        return json.dumps(response_data)

    async def handle_request(self, request: Any) -> Optional[Any]:
        """Processes a parsed JSON-RPC request (dict) or list of requests (batch)."""
        if isinstance(request, list):
            # Batch request
            if not request:
                return self._make_error_response(INVALID_REQUEST, "Invalid Request: Empty batch.", None, None)
            responses = []
            for req in request:
                res = await self._process_single_request(req)
                if res is not None:
                    responses.append(res)
            return responses
        
        return await self._process_single_request(request)

    async def _process_single_request(self, req: Any) -> Optional[Dict[str, Any]]:
        """Processes a single JSON-RPC request dictionary."""
        import inspect
        # 1. Log the incoming tool call
        logger.info(f"JSON-RPC Request: {json.dumps(req)}")
        
        req_id = None
        try:
            # Validate JSON-RPC structure
            if not isinstance(req, dict):
                raise JSONRPCError(INVALID_REQUEST, "Invalid Request: Must be a JSON object.")
            
            if req.get("jsonrpc") != "2.0":
                raise JSONRPCError(INVALID_REQUEST, "Invalid Request: 'jsonrpc' version must be exactly '2.0'.")
            
            if "method" not in req:
                raise JSONRPCError(INVALID_REQUEST, "Invalid Request: Missing 'method' field.")
            
            method_name = req["method"]
            if not isinstance(method_name, str):
                raise JSONRPCError(INVALID_REQUEST, "Invalid Request: 'method' must be a string.")
            
            req_id = req.get("id")
            
            # Fetch params
            params = req.get("params", {})
            if not isinstance(params, (dict, list)):
                raise JSONRPCError(INVALID_REQUEST, "Invalid Request: 'params' must be structured as a JSON Object or Array.")

            # Look up method
            if method_name not in self.methods:
                raise JSONRPCError(METHOD_NOT_FOUND, f"Method not found: '{method_name}'")

            func = self.methods[method_name]

            # Execute method (supports sync and async functions)
            try:
                if isinstance(params, dict):
                    res_or_coro = func(**params)
                else:  # list
                    res_or_coro = func(*params)
                
                if inspect.iscoroutine(res_or_coro):
                    result = await res_or_coro
                else:
                    result = res_or_coro
            except TypeError as te:
                # Catch incorrect argument counts
                raise JSONRPCError(INVALID_PARAMS, f"Invalid params: {te}")
            except Exception as e:
                logger.error(f"Error executing method '{method_name}': {e}", exc_info=True)
                raise JSONRPCError(INTERNAL_ERROR, f"Internal error during execution: {str(e)}")

            # Formulate response (if not a notification - notifications have id=None/omitted)
            if req_id is not None:
                response = {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": req_id
                }
                logger.info(f"JSON-RPC Response (Success): {json.dumps(response)}")
                return response
            else:
                logger.info("JSON-RPC Notification processed (no response returned).")
                return None

        except JSONRPCError as je:
            response = self._make_error_response(je.code, je.message, je.data, req_id)
            logger.warning(f"JSON-RPC Response (Error): {json.dumps(response)}")
            return response
        except Exception as e:
            response = self._make_error_response(INTERNAL_ERROR, f"Internal error: {str(e)}", None, req_id)
            logger.error(f"JSON-RPC Response (Unhandled Error): {json.dumps(response)}")
            return response

    def _make_error_response(self, code: int, message: str, data: Any, req_id: Any) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message,
                **({"data": data} if data is not None else {})
            },
            "id": req_id
        }
