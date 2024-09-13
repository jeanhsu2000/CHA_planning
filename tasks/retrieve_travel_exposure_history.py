from typing import Any, List
from tasks.task import BaseTask


class RetrieveTravelExposureHistory(BaseTask):
    """
    **Description:**

        This task retrieves the patient's recent travel history and exposure to individuals with infections or respiratory symptoms.
    """

    name: str = "retrieve_travel_exposure_history"
    chat_name: str = "RetrieveTravelExposureHistory"
    description: str = "Fetches the patient's recent travel history and exposure to individuals with infections or respiratory symptoms."
    dependencies: List[str] = []
    inputs: List[str] = ["Patient ID."]
    outputs: List[str] = [
        "Returns a json object with travel history and recent exposure data. For example: {'travel_history': 'Traveled to high-risk area', 'recent_exposure': 'Close contact with respiratory symptoms'}"
    ]

    def _execute(self, inputs: List[Any] = None) -> str:
        patient_id = inputs[0]
        # Simulated retrieval of travel and exposure history
        result = {
            "travel_history": "Traveled to high-risk area 7 days ago",
            "recent_exposure": "Contact with sick individuals 3 days ago",
        }
        return result

    def explain(self) -> str:
        return "This task retrieves the patient's recent travel history and possible exposure to individuals with any infections or respiratory symptoms."