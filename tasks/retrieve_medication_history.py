from typing import Any, List
from tasks.task import BaseTask


class RetrieveMedicationHistory(BaseTask):
    """
    **Description:**

        This task retrieves the patient's medication history, including medication names, dosages, and dates of use.
    """

    name: str = "retrieve_medication_history"
    chat_name: str = "RetrieveMedicationHistory"
    description: str = "Fetches the patient's medication history, including names, dosages, and dates of use."
    dependencies: List[str] = []
    inputs: List[str] = ["Patient ID."]
    outputs: List[str] = [
        "Returns a json object with the medication history. For example: {'medications': [{'name': 'Lisinopril', 'dosage': '10mg', 'start_date': '2022-01-10', 'end_date': '2022-06-10'}, {'name': 'Metformin', 'dosage': '500mg', 'start_date': '2023-02-01', 'ongoing': True}]}"
    ]

    def _execute(self, inputs: List[Any] = None) -> str:
        patient_id = inputs[0]
        # Simulated retrieval of medication history
        result = {
            "medications": [
                {
                    "name": "Lisinopril",
                    "dosage": "10mg",
                    "start_date": "2022-01-10",
                    "end_date": "2022-06-10"
                },
                {
                    "name": "Metformin",
                    "dosage": "500mg",
                    "start_date": "2023-02-01",
                    "ongoing": True
                }
            ]
        }
        return result

    def explain(self) -> str:
        return "This task retrieves the patient's medication history, including medication names, dosages, and dates of use."
