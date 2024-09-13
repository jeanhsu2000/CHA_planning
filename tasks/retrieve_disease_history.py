from typing import Any, List, Dict
from tasks.task import BaseTask


class RetrieveDiseaseHistory(BaseTask):
    """
    **Description:**

        This task retrieves the patient's history of medical conditions, including allergies, infections, chronic diseases such as asthma, heart issues, and others. 
        It also fetches the corresponding treatments provided for these conditions, including medications, therapies, surgeries, or lifestyle changes.
    """

    name: str = "retrieve_disease_history"
    chat_name: str = "RetrieveDiseaseHistory"
    description: str = "Fetches the patient's history of diseases and conditions, including allergies, infections, asthma, heart disease, and others, along with corresponding treatments."
    dependencies: List[str] = []
    inputs: List[str] = ["Patient ID."]
    outputs: List[str] = [
        "Returns a json object with disease history and treatments. For example: {'conditions': [{'type': 'Asthma', 'date_diagnosed': '2010-04-15', 'treatments': [{'name': 'Inhaler', 'type': 'Medication', 'start_date': '2010-04-16', 'end_date': None}, {'name': 'Albuterol', 'type': 'Medication', 'start_date': '2010-04-16', 'end_date': '2020-04-16'}]}, {'type': 'Heart Disease', 'date_diagnosed': '2015-06-20', 'treatments': [{'name': 'Statins', 'type': 'Medication', 'start_date': '2015-06-21', 'end_date': None}, {'name': 'Bypass Surgery', 'type': 'Surgery', 'date': '2017-08-15'}]}]}"
    ]

    def _execute(self, inputs: List[Any] = None) -> Dict[str, Any]:
        patient_id = inputs[0]
        # Simulated retrieval of disease history with treatments
        result = {
            "conditions": [
                {
                    "type": "Asthma",
                    "date_diagnosed": "2010-04-15",
                    "treatments": [
                        {
                            "name": "Inhaler",
                            "type": "Medication",
                            "start_date": "2010-04-16",
                            "end_date": None
                        },
                        {
                            "name": "Albuterol",
                            "type": "Medication",
                            "start_date": "2010-04-16",
                            "end_date": "2020-04-16"
                        }
                    ]
                },
                {
                    "type": "Heart Disease",
                    "date_diagnosed": "2015-06-20",
                    "treatments": [
                        {
                            "name": "Statins",
                            "type": "Medication",
                            "start_date": "2015-06-21",
                            "end_date": None
                        },
                        {
                            "name": "Bypass Surgery",
                            "type": "Surgery",
                            "date": "2017-08-15"
                        }
                    ]
                },
                {
                    "type": "COVID-19",
                    "date_diagnosed": "2022-01-10",
                    "treatments": [
                        {
                            "name": "Antiviral Medication",
                            "type": "Medication",
                            "start_date": "2022-01-15",
                            "end_date": "2022-01-25"
                        }
                    ]
                },
                {
                    "type": "Allergies",
                    "trigger": "Pollen",
                    "date_diagnosed": "2008-03-12",
                    "treatments": [
                        {
                            "name": "Antihistamines",
                            "type": "Medication",
                            "start_date": "2008-03-13",
                            "end_date": None
                        },
                        {
                            "name": "Allergy Shots",
                            "type": "Therapy",
                            "start_date": "2008-04-01",
                            "end_date": "2012-04-01"
                        }
                    ]
                }
            ]
        }
        return result

    def explain(self) -> str:
        return ("This task retrieves the patient's disease history, including allergies, infections, asthma, heart disease, "
                "and other chronic conditions, along with the corresponding treatments. Treatments may include medications, surgeries, "
                "therapies, and lifestyle changes. The task helps provide a comprehensive medical overview of the patient's history.")

