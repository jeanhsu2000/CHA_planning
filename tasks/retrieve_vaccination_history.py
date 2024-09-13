from typing import Any, List
from tasks.task import BaseTask


class RetrieveVaccinationHistory(BaseTask):
    """
    **Description:**

        This task retrieves the patient's vaccination history, including vaccine types, dates, and last vaccination received.
    """

    name: str = "retrieve_vaccination_history"
    chat_name: str = "RetrieveVaccinationHistory"
    description: str = "Fetches the patient's vaccination history, including vaccine types and administration dates."
    dependencies: List[str] = []
    inputs: List[str] = ["Patient ID."]
    outputs: List[str] = [
        "Returns a json object with the vaccination history. For example: {'vaccines': [{'vaccine_name': 'Influenza', 'date': '2022-10-15'}, {'vaccine_name': 'COVID-19', 'date': '2023-05-12'}]}"
    ]

    def _execute(self, inputs: List[Any] = None) -> str:
        patient_id = inputs[0]
        # Simulated retrieval of vaccination history
        result = {
            "vaccines": [
                {"vaccine_name": "Influenza", "date": "2022-10-15"},
                {"vaccine_name": "COVID-19", "date": "2023-05-12"},
            ]
        }
        return result

    def explain(self) -> str:
        return "This task retrieves the patient's vaccination history, including types of vaccines and administration dates."