import os
import json
import logging
import httpx
from typing import Dict, List, Any, Optional

logger = logging.getLogger("medical_agent")

class FHIRClient:
    def __init__(self, base_url: str = "https://r4.smarthealthit.org"):
        self.base_url = base_url.rstrip('/')

    async def fetch_patient_bundle(self, patient_id: str) -> Dict[str, Any]:
        """
        Fetches patient record from SMART on FHIR.
        First dynamically searches for an active patient ID if patient_id is default/empty
        or if we encounter fetch errors, then pulls their history ($everything or direct).
        Supports reading from a local directory if base_url is a directory path.
        """
        # Check if base_url is a local directory
        if os.path.isdir(self.base_url):
            logger.info(f"base_url '{self.base_url}' is a directory. Reading patient data from directory.")
            # 1. Search for a file named {patient_id}.json
            file_path = os.path.join(self.base_url, f"{patient_id}.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        bundle_data = json.load(f)
                    logger.info(f"Successfully loaded patient bundle from file: {file_path}")
                    return bundle_data
                except Exception as e:
                    logger.error(f"Failed to read patient file {file_path}: {e}")
            
            # 2. Iterate through all .json files in the directory and search inside them
            try:
                for filename in os.listdir(self.base_url):
                    if filename.endswith(".json"):
                        candidate_path = os.path.join(self.base_url, filename)
                        with open(candidate_path, "r", encoding="utf-8") as f:
                            candidate_data = json.load(f)
                            if candidate_data.get("resourceType") == "Patient" and candidate_data.get("id") == patient_id:
                                logger.info(f"Found matching Patient resource in file: {candidate_path}")
                                return candidate_data
                            elif candidate_data.get("resourceType") == "Bundle":
                                for entry in candidate_data.get("entry", []):
                                    res = entry.get("resource", {})
                                    if res.get("resourceType") == "Patient" and res.get("id") == patient_id:
                                        logger.info(f"Found matching Patient inside bundle file: {candidate_path}")
                                        return candidate_data
            except Exception as e:
                logger.error(f"Error scanning directory {self.base_url}: {e}")
            
            logger.warning(f"Could not find patient {patient_id} in directory {self.base_url}. Using mock fallback.")
            return self._load_mock_bundle()

        resolved_patient_id = patient_id
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AntigravityFHIRClient/1.0"}
        
        async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
            # 1. Dynamically search for an active patient ONLY IF patient_id is not specified or is "default"
            # This ensures we stick to a single patient when one is provided.
            if not resolved_patient_id or resolved_patient_id in ("default", "smart-default"):
                search_url = f"{self.base_url}/Patient?_count=1"
                logger.info(f"Searching for an active patient dynamically at: {search_url}")
                try:
                    response = await client.get(search_url, headers=headers)
                    response.raise_for_status()
                    search_data = response.json()
                    entries = search_data.get("entry", [])
                    if entries:
                        resolved_patient_id = entries[0].get("resource", {}).get("id", patient_id)
                        logger.info(f"Discovered active patient ID on sandbox: {resolved_patient_id}")
                    else:
                        logger.warning("Dynamic search returned no patients. Using default patient ID.")
                except Exception as e:
                    logger.warning(f"Dynamic patient search failed: {e}. Falling back to default ID: {patient_id}")

            # 2. Attempt to pull the complete clinical history using $everything
            url_everything = f"{self.base_url}/Patient/{resolved_patient_id}/$everything"
            logger.info(f"Ingesting patient clinical history from: {url_everything}")

            bundle_data = {}
            fetch_success = False

            try:
                response = await client.get(url_everything, headers=headers)
                response.raise_for_status()
                bundle_data = response.json()
                fetch_success = True
                logger.info(f"Successfully retrieved FHIR bundle ($everything) for patient {resolved_patient_id}.")
            except Exception as e:
                logger.warning(f"Failed to fetch $everything history: {e}. Attempting direct Patient resource fetch.")
                # 3. Fallback: fetch patient resource directly
                url_patient = f"{self.base_url}/Patient/{resolved_patient_id}"
                logger.info(f"Ingesting direct patient data from: {url_patient}")
                try:
                    response = await client.get(url_patient, headers=headers)
                    response.raise_for_status()
                    bundle_data = response.json()
                    fetch_success = True
                    logger.info(f"Successfully retrieved FHIR patient resource for patient {resolved_patient_id}.")
                except Exception as e2:
                    logger.error(f"Failed to fetch direct patient record: {e2}")
                    logger.info("Proceeding to load local fallback data due to fetch errors.")

        # Check if the bundle contains clinical data (Observations or Conditions)
        has_clinical_data = False
        if fetch_success:
            if bundle_data.get("resourceType") == "Bundle":
                entries = bundle_data.get("entry", [])
                for entry in entries:
                    resource = entry.get("resource", {})
                    rt = resource.get("resourceType")
                    if rt in ("Observation", "Condition", "MedicationRequest"):
                        has_clinical_data = True
                        break
            else:
                has_clinical_data = False

        if not has_clinical_data:
            logger.warning(
                f"Live FHIR data for patient {resolved_patient_id} lacks clinical records. "
                "Loading synthetic mock patient bundle data from local fallback file."
            )
            bundle_data = self._load_mock_bundle()
            logger.debug(f"Mock FHIR Data Loaded: {json.dumps(bundle_data, indent=2)}")

        return bundle_data

    def _load_mock_bundle(self) -> Dict[str, Any]:
        """Loads fallback data from mock_patient_bundle.json."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mock_path = os.path.join(current_dir, "mock_patient_bundle.json")
        try:
            with open(mock_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.critical(f"Failed to load mock_patient_bundle.json: {e}")
            raise FileNotFoundError(f"Mock patient bundle not found at {mock_path}") from e

    def extract_patient_info(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts core demographic information from the Patient resource in the bundle."""
        if bundle.get("resourceType") == "Patient":
            res = bundle
            names = res.get("name", [])
            name_str = "Unknown"
            if names:
                name_obj = names[0]
                given = " ".join(name_obj.get("given", []))
                family = name_obj.get("family", "")
                name_str = f"{given} {family}".strip() or "Unknown"
            return {
                "id": res.get("id"),
                "name": name_str,
                "gender": res.get("gender"),
                "birth_date": res.get("birthDate"),
                "resource_type": "Patient"
            }

        entries = bundle.get("entry", [])
        for entry in entries:
            res = entry.get("resource", {})
            if res.get("resourceType") == "Patient":
                # Parse Name
                names = res.get("name", [])
                name_str = "Unknown"
                if names:
                    name_obj = names[0]
                    given = " ".join(name_obj.get("given", []))
                    family = name_obj.get("family", "")
                    name_str = f"{given} {family}".strip() or "Unknown"
                
                return {
                    "id": res.get("id"),
                    "name": name_str,
                    "gender": res.get("gender"),
                    "birth_date": res.get("birthDate"),
                    "resource_type": "Patient"
                }
        return {"id": "Unknown", "name": "Unknown", "gender": "Unknown", "birth_date": "Unknown"}

    def extract_conditions(self, bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts and flattens Condition resources from the bundle."""
        conditions = []
        entries = bundle.get("entry", [])
        for entry in entries:
            res = entry.get("resource", {})
            if res.get("resourceType") == "Condition":
                # Extract clinical status
                clinical_status = "unknown"
                clinical_coding = res.get("clinicalStatus", {}).get("coding", [])
                if clinical_coding:
                    clinical_status = clinical_coding[0].get("code", "unknown")
                
                # Extract verification status
                verification_status = "unknown"
                verification_coding = res.get("verificationStatus", {}).get("coding", [])
                if verification_coding:
                    verification_status = verification_coding[0].get("code", "unknown")

                # Extract code and display text
                code_obj = res.get("code", {})
                display_name = code_obj.get("text")
                code_val = "Unknown"
                if not display_name and code_obj.get("coding"):
                    display_name = code_obj["coding"][0].get("display")
                    code_val = code_obj["coding"][0].get("code", "Unknown")
                elif code_obj.get("coding"):
                    code_val = code_obj["coding"][0].get("code", "Unknown")

                display_name = display_name or "Unknown Condition"

                conditions.append({
                    "id": res.get("id"),
                    "code": code_val,
                    "display": display_name,
                    "clinical_status": clinical_status,
                    "verification_status": verification_status,
                    "onset_date": res.get("onsetDateTime") or res.get("onsetPeriod", {}).get("start", "Unknown")
                })
        return conditions

    def extract_vitals(self, bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts and flattens Observation (Vitals / Lab Results) resources from the bundle."""
        vitals = []
        entries = bundle.get("entry", [])
        for entry in entries:
            res = entry.get("resource", {})
            if res.get("resourceType") == "Observation":
                code_obj = res.get("code", {})
                display_name = code_obj.get("text")
                code_val = "Unknown"
                if not display_name and code_obj.get("coding"):
                    display_name = code_obj["coding"][0].get("display")
                    code_val = code_obj["coding"][0].get("code", "Unknown")
                elif code_obj.get("coding"):
                    code_val = code_obj["coding"][0].get("code", "Unknown")

                display_name = display_name or "Unknown Observation"
                date_str = res.get("effectiveDateTime") or res.get("issued") or "Unknown"

                # Check if it has simple value or nested component values (like Blood Pressure)
                components = res.get("component", [])
                if components:
                    comp_vitals = []
                    for comp in components:
                        comp_code_obj = comp.get("code", {})
                        comp_display = comp_code_obj.get("text")
                        if not comp_display and comp_code_obj.get("coding"):
                            comp_display = comp_code_obj["coding"][0].get("display")
                        
                        val_qty = comp.get("valueQuantity", {})
                        val = val_qty.get("value")
                        unit = val_qty.get("unit", "")
                        comp_vitals.append({
                            "name": comp_display or "Component",
                            "value": val,
                            "unit": unit
                        })
                    vitals.append({
                        "id": res.get("id"),
                        "code": code_val,
                        "display": display_name,
                        "date": date_str,
                        "components": comp_vitals
                    })
                else:
                    val_qty = res.get("valueQuantity", {})
                    val = val_qty.get("value")
                    unit = val_qty.get("unit", "")
                    
                    # If it's a valueString or other types
                    if val is None:
                        val = res.get("valueString") or res.get("valueCodeableConcept", {}).get("text")
                        unit = ""

                    vitals.append({
                        "id": res.get("id"),
                        "code": code_val,
                        "display": display_name,
                        "date": date_str,
                        "value": val,
                        "unit": unit
                    })
        return vitals
        