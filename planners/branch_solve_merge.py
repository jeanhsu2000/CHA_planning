"""
Heavily borrowed from langchain: https://github.com/langchain-ai/langchain/
"""
import re
from typing import Any
from typing import List
from typing import ClassVar

from planners.action import Action
from planners.action import PlanFinish
from planners.planner import BasePlanner
from tasks.task import OutputType
from tasks.internals.ask_user import AskUser


class BranchSolveMergePlanner(BasePlanner):
    """
    **Description:**

        This class implements the Branch-Solve-Merge planner, which inherits from the BasePlanner base class.
        Branch-Solve-Merge employs dynamic problem decomposition and solution synthesis strategies by creating 
        parallel problem-solving branches, where each branch independently explores distinct solution pathways.
        The planner then merges these branches to construct the final solution.
        `Paper <https://arxiv.org/abs/2310.15123>`_
    """

    summarize_prompt: bool = True
    max_tokens_allowed: int = 10000

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _planner_type(self):
        return "zero-shot-react-planner"

    @property
    def _planner_model(self):
        return self.llm_model

    @property
    def _response_generator_model(self):
        return self.llm_model

    @property
    def _stop(self) -> List[str]:
        return ["Wait"]

    @property
    def _shorten_prompt(self):
        return (
            "Summarize the following text. Make sure to keep the main ideas "
            "and objectives in the summary. Keep the links "
            "exactly as they are: "
            "{chunk}"
        )

    @property
    def _planner_prompt(self):
        return [
            """As a knowledgeable and empathetic health assistant, your primary objective is to provide the user with precise and valuable \
information regarding their health and well-being. Utilize the available tools effectively to answer health-related queries. \
Here are the tools at your disposal:
{tool_names}

The following is the format of the information provided:
MetaData: this contains the name of data files of different types like image, audio, video, and text. You can pass these files to tools when needed.

Use the tools and provided information, first suggest three possible diagnoses for the patient based on their symptoms and medical history. \
Give an detailed explanation consisting of sequences of tools to properly come up with the diagnoses to answer the user query.The tools constraints should be always satisfied. \
For each diagnosis, list the steps, tests, or treatments you would recommend to confirm or address it. \
Finally, combine the results from all the diagnoses into a comprehensive treatment plan or next steps.
Make sure the final response is comprehensive enough and use proper tools. 
**You should stick to the provided tools and never use any other tasks.** Never try to load files as your strategy steps.

Write the detailed tool executions step by step. Start your final diagnosis with
'Final diagnosis:'.

Begin!

MetaData:
{meta}
=========================
{previous_actions}
=========================
{history}
USER: {input}
CHA:
""",
            """
{strategy}
=========================
{previous_actions}
=========================
Tools:
{tool_names}
=========================
MetaData:
{meta}
=========================
{history}
USER: {input}

You are skilled python programmer that can solve problems and convert them into python codes. \
Using the selected final strategy mentioned in the 'Decision:
', create a python code inside a ```python ``` block that outlines a sequence of steps using the Tools. \
assume that there is an **self.execute_task** function that can execute the tools in it. The execute_task \
recieves task name and an array of the inputs and returns the result. Make sure that you always pass and array as second argument. \
You can call tools like this: \
**task_result = self.execute_task('tool_name', ['input_1', 'input2', ...])**. \
The flow should utilize this style representing the tools available. Make sure all the execute_task calls outputs are stored in a variable.\
If a step's output is required as input for a subsequent step, ensure the python code captures this dependency clearly. \
The output variables should directly passed as inputs with no changes in the wording.
If the tool input is a datapipe only put the variable as the input. \
For each tool, include necessary parameters directly without any names and assume each will return an output. \
The outputs' description are provided for each Tool individually. Make sure you use the directives when passing the outputs.

Question: {input}
""",
        ]
    
    # Expanded category mapping for healthcare-related queries
    methodology_mapping: ClassVar[dict] = {
        "differential_diagnosis": {
            "instructions": "Use multimodal patient data such as medical records, imaging, and patient-reported outcomes to suggest potential diagnoses. Formulate mutually exclusive diagnostic sets using Branch-Solve-Merge.",
            "methodology": "Diagnostic Hypothesis Generation",
            "response_start": "Possible diagnoses based on patient data:",
            "follow_up_instructions": "Generate follow-up questions to clarify symptoms, patient history, and potential confounding factors."
        },
        "treatment_options": {
            "instructions": "Using the patient's data, generate personalized treatment plans that consider lifestyle, preferences, and multimodal health data.",
            "methodology": "Goal-Oriented Treatment Planning",
            "response_start": "Personalized treatment options:",
            "follow_up_instructions": "Ask questions that refine patient goals and preferences for treatment."
        },
        "preventive_measures": {
            "instructions": "Recommend preventive strategies based on lifestyle factors, family history, and wearable data.",
            "methodology": "Preventive Strategy Decomposition",
            "response_start": "Preventive measures tailored to your health:",
            "follow_up_instructions": "Inquire about lifestyle and risk factors."
        },
        "prognosis": {
            "instructions": "Provide a detailed prognosis using longitudinal data from the patient's health history.",
            "methodology": "Longitudinal Data Forecasting",
            "response_start": "Prognosis based on patient history:",
            "follow_up_instructions": "Ask questions about current symptoms and treatment history."
        },
        "test_interpretation": {
            "instructions": "Interpret test results using the patient's multimodal health data and explain the findings clearly.",
            "methodology": "Data Interpretation Framework",
            "response_start": "Test result interpretation:",
            "follow_up_instructions": "Ask clarifying questions about symptoms and test conditions."
        },
        "rehabilitation": {
            "instructions": "Develop a personalized rehabilitation plan based on the patient's data, including therapy suggestions.",
            "methodology": "Rehabilitation Pathway Planning",
            "response_start": "Personalized rehabilitation plan:",
            "follow_up_instructions": "Ask about functional limitations and therapy responses."
        },
        "mental_health_support": {
            "instructions": "Provide a comprehensive mental health support plan, ensuring alignment with the patient's mental health goals.",
            "methodology": "Mental Health Management Planning",
            "response_start": "Personalized mental health support plan:",
            "follow_up_instructions": "Ask about mental health history and current concerns."
        },
        "lifestyle_and_wellness": {
            "instructions": "Recommend personalized wellness strategies based on the patient’s long-term health goals and current data.",
            "methodology": "Lifestyle Optimization",
            "response_start": "Wellness recommendations tailored for you:",
            "follow_up_instructions": "Ask about lifestyle habits and preferences for wellness planning."
        },
        "medication_management": {
            "instructions": "Generate personalized medication management guidance considering the patient's current medications and preferences.",
            "methodology": "Medication Management Protocol",
            "response_start": "Medication management plan:",
            "follow_up_instructions": "Ask about current medications and adherence."
        },
        "follow_up_care": {
            "instructions": "Recommend a follow-up care plan tailored to the patient’s evolving health status.",
            "methodology": "Longitudinal Follow-Up Planning",
            "response_start": "Follow-up care recommendations:",
            "follow_up_instructions": "Inquire about current symptoms and health status."
        },
        "chronic_disease_management": {
            "instructions": "Develop a chronic disease management plan using patient preferences, wearable data, and test results.",
            "methodology": "Chronic Disease Pathway Management",
            "response_start": "Chronic disease management plan:",
            "follow_up_instructions": "Ask about the patient's condition and current treatments."
        },
        "screening_recommendations": {
            "instructions": "Generate personalized screening recommendations based on the patient's risk factors and lifestyle.",
            "methodology": "Risk-Based Screening Strategy",
            "response_start": "Screening recommendations:",
            "follow_up_instructions": "Ask about family history and previous screening results."
        },
        "public_health_and_safety": {
            "instructions": "Provide patient-specific public health recommendations, including guidance on vaccinations and infection prevention.",
            "methodology": "Public Health Guidance",
            "response_start": "Public health and safety recommendations:",
            "follow_up_instructions": "Ask about travel plans and vaccination status."
        },
        "health_monitoring": {
            "instructions": "Suggest personalized health monitoring protocols using wearable data and check-ins.",
            "methodology": "Health Monitoring Protocol Design",
            "response_start": "Personalized health monitoring plan:",
            "follow_up_instructions": "Ask about wearable devices and health goals."
        },
        "patient_education": {
            "instructions": "Generate patient-friendly educational materials explaining their condition and treatment options.",
            "methodology": "Patient Education Framework",
            "response_start": "Patient education material:",
            "follow_up_instructions": "Ask about the patient's understanding of their condition and treatments."
        },
        "care_coordination": {
            "instructions": "Coordinate care across multiple providers, ensuring that all information is shared effectively.",
            "methodology": "Care Coordination Strategy",
            "response_start": "Care coordination plan:",
            "follow_up_instructions": "Ask about the patient's current providers and any care challenges."
        }
    }

    def normalize_category(self, input_category: str) -> str:
        """
        Normalize the input category by removing spaces and converting to lowercase.
        """
        return input_category.lower().replace(" ", "_").strip()
    
    def categorize_inquiry(self, inquiry: str) -> str:
        """
        Categorizes the user query into one of the defined healthcare categories.
        """
        prompt = f"""
        Please categorize the following medical question into one of the following categories:
        - Differential Diagnosis
        - Treatment Options
        - Preventive Measures
        - Prognosis
        - Test Interpretation
        - Rehabilitation
        - Mental Health Support
        - Lifestyle and Wellness
        - Medication Management
        - Follow-Up Care
        - Chronic Disease Management
        - Screening Recommendations
        - Public Health and Safety
        - Health Monitoring
        - Patient Education
        - Care Coordination
        
        Question: {inquiry}

        Response Format:
        {{
            "category": "differential_diagnosis"
        }}
        """
        response = self._planner_model.generate(query=prompt)
        category_match = re.search(r'"category":\s*"([^"]+)"', response)
        if category_match:
            raw_category = category_match.group(1).lower().strip()
        else:
            return "uncategorized"  # Default if no match

        # Normalize the raw category to map to the correct key
        normalized_category = self.normalize_category(raw_category)

        # Check if the normalized category is an exact match
        if normalized_category in self.methodology_mapping:
            return normalized_category
        
        return "uncategorized"

    def task_descriptions(self):
        return "".join(
            [
                (
                    "\n-----------------------------------\n"
                    f"**{task.name}**: {task.description}"
                    "\nThis tool have the following outputs:\n"
                    + "\n".join(task.outputs)
                    + (
                        "\n- The result of this tool will be stored in the datapipe."
                        if task.output_type == OutputType.DATAPIPE
                        else ""
                    )
                    + "\n-----------------------------------\n"
                )
                if not task.executor_task
                else ""
                for task in self.available_tasks
            ]
        )

    def divide_text_into_chunks(
        self,
        input_text: str = "",
        max_tokens: int = 10000,
    ) -> List[str]:
        """
        Generate a response based on the input prefix, query, and thinker (task planner).

        Args:
            input_text (str): the input text (e.g., prompt).
            max_tokens (int): Maximum number of tokens allowed.
        Return:
            chunks(List): List of string variables
        """
        # 1 token ~= 4 chars in English
        chunks = [
            input_text[i : i + max_tokens * 4]
            for i in range(0, len(input_text), max_tokens * 4)
        ]
        return chunks

    def generate_scratch_pad(
        self, previous_actions: List[str] = None, **kwargs: Any
    ):
        if previous_actions is None:
            previous_actions = []

        agent_scratchpad = ""
        if len(previous_actions) > 0:
            agent_scratchpad = "\n".join(
                [f"\n{action}" for action in previous_actions]
            )
        # agent_scratchpad
        if (
            self.summarize_prompt
            and len(agent_scratchpad) / 4 > self.max_tokens_allowed
        ):
            # Shorten agent_scratchpad
            chunks = self.divide_text_into_chunks(
                input_text=agent_scratchpad,
                max_tokens=self.max_tokens_allowed,
            )
            agent_scratchpad = ""
            kwargs["max_tokens"] = min(
                2000, int(self.max_tokens_allowed / len(chunks))
            )
            for chunk in chunks:
                prompt = self._shorten_prompt.replace(
                    "{chunk}", chunk
                )
                chunk_summary = (
                    self._response_generator_model.generate(
                        query=prompt, **kwargs
                    )
                )
                agent_scratchpad += chunk_summary + " "

    def assess_ambiguity(self, original_query: str, category: str) -> float:
        """
        Assess the ambiguity level of the query, generalized across all categories.
        This function evaluates factors such as vagueness, missing info, and overlap based on the query type.
        """

        # Symptom Overlap Ambiguity or Query Ambiguity
        prompt_overlap_or_ambiguity = f"""
        You are a medical assistant specializing in healthcare queries across categories such as diagnosis, treatment, and prognosis.
        For the following query:
        "{original_query}"
        Evaluate how clear or vague the query is based on its category "{category}". Does it contain ambiguous or overlapping details that need further clarification? 

        Return a value between 0.0 (clear and specific) and 1.0 (very ambiguous).
        """
        overlap_ambiguity_response = self._planner_model.generate(query=prompt_overlap_or_ambiguity)
        overlap_ambiguity_score = float(re.search(r"(\d+\.\d+)", overlap_ambiguity_response).group(1))

        # Missing Critical Information (generalized across all categories)
        prompt_missing_info = f"""
        Assess whether critical information is missing from the following healthcare query:
        "{original_query}"
        Return a score between 0.0 (nothing critical missing) and 1.0 (critical info missing).
        """
        missing_info_response = self._planner_model.generate(query=prompt_missing_info)
        missing_info_score = float(re.search(r"(\d+\.\d+)", missing_info_response).group(1))

        # Combine scores to get an overall ambiguity score
        total_ambiguity_score = (overlap_ambiguity_score + missing_info_score) / 2
        return total_ambiguity_score

    def determine_branch_count(self, ambiguity_score: float, original_query: str) -> int:
        """
        Dynamically decide the number of branches based on both the ambiguity score and the complexity of the query itself.
        The more complex and ambiguous the query, the more branches might be needed.
        """
        # Use ChatGPT to assess the complexity of the query directly
        prompt_complexity = f"""
        You are an assistant determining the complexity of a healthcare query.
        Evaluate the complexity of this query: "{original_query}"
        Based on its complexity, suggest an appropriate number of diagnostic or treatment paths (branches) to explore.

        The query's ambiguity score is {ambiguity_score}. Factor this score into your decision for how many follow-up paths (branches) to explore. Return only the number of branches you think is appropriate.
        """
        complexity_response = self._planner_model.generate(query=prompt_complexity)
        branch_count = int(re.search(r"(\d+)", complexity_response).group(1))

        return branch_count

    def adjust_question_detail(self, ambiguity_score: float) -> str:
        """
        Adjust the level of detail for follow-up questions based on ambiguity score.
        - High ambiguity → broad questions.
        - Low ambiguity → more detailed, specific questions.
        """
        if ambiguity_score > 0.75:
            return "Start with broad, general questions to gather more context."
        elif ambiguity_score > 0.5:
            return "Ask moderately detailed questions, ensuring that you gather comprehensive responses."
        else:
            return "Focus on specific, detail-oriented questions to clarify key aspects."


    def generate_bsm_template(self, category: str, original_query: str, tool_descriptions: str) -> str:
        """
        Generates the Branch-Solve-Merge workflow, integrating ambiguity assessment and dynamic branching.
        """
        # Assess ambiguity and determine dynamic branch count
        ambiguity_score = self.assess_ambiguity(original_query, category)
        branch_count = self.determine_branch_count(ambiguity_score, original_query)
        question_detail_level = self.adjust_question_detail(ambiguity_score)

        # Display the ambiguity score for the user
        ambiguity_assessment_output = f"Ambiguity Score: {ambiguity_score:.2f}/1.0 (Higher means more ambiguity)\n"
        print(ambiguity_assessment_output)  # Display to the user or log
        print("Question detail level: ", question_detail_level)

        methodology_info = self.methodology_mapping.get(category, {})
        instructions = methodology_info.get("instructions", "")
        response_start = methodology_info.get("response_start", "")
        follow_up_instructions = methodology_info.get("follow_up_instructions", "")
        methodology = methodology_info.get("methodology", "Branch-Solve-Merge")

        # Step 1: Generate the Branch portion based on the category and methodology
        methodology = self.methodology_mapping[category]["methodology"]
        branch_prompt = f"""
        You are a highly knowledgeable and empathetic healthcare assistant using the **{methodology}** methodology within a **Branch-Solve-Merge (BSM) workflow**. Your task is to address the following healthcare query by breaking it down into structured, distinct paths. The BSM methodology will help you ensure that each branch is well-defined and tailored to the nature of the query.

        ### Context of the BSM Workflow:
        - **Branch**: In this step, you will divide the query into distinct branches based on the patient's medical information or the nature of the query. Each branch represents a potential diagnosis, treatment option, preventive measure, or other health strategy, depending on the category of the query.
        - **Solve**: Follow-up questions will be created in the next step to gather more information or clarify key aspects of the branches.
        - **Merge**: After sufficient information has been gathered, the branches will be merged into a comprehensive response.

        ### Inputs:
        - **Category**: {category}
        - **Original Query**: {original_query}
        - **Original Instructions**: {instructions}
        - **Response Start**: {response_start}

        ### Your Task (Branch Step):
        In this step, as a healthcare assistant, your task is t break down the query into {branch_count} distinct paths (branches) that address the query according to the **{category}** methodology. 
        Each branch should reflect a different approach relevant to the patient's situation or the nature of the inquiry (e.g., diagnosis, treatment, prognosis, preventive care, etc.).

        - For each branch, outline key areas where additional information will be required. These areas will guide the creation of follow-up questions in the **Solve** step.
        - The branches should be **distinct**, **clinically relevant**, and **aligned with the query's focus** (e.g., diagnostic possibilities, treatment options, prevention strategies, etc.).
        - You should aim for **a balance** between oversimplifying (too few branches) and overcomplicating (too many branches) the query. The number of branches should be guided by the query's complexity.

        Now, generate the **Branch** portion of the BSM workflow based on the category "{category}".
        """

        branch_output = self._planner_model.generate(query=branch_prompt).strip()

        # Step 2: Generate the Solve portion, ensuring the follow-up questions are in the specified format
        solve_prompt = f"""
        You are a highly knowledgeable and empathetic healthcare assistant continuing the **{methodology}** methodology within a **Branch-Solve-Merge (BSM) workflow**. In this step, your task is to generate detailed and relevant follow-up questions based on the branches identified in the previous step.

        ### Context of the BSM Workflow:
        - **Branch**: The query has been divided into distinct paths based on the patient's information or the nature of the query (e.g., diagnosis, treatment, preventive care, etc.).
        - **Solve**: Now, your role is to create detailed follow-up questions for each branch to gather further information or clarify key areas.
        - **Merge**: After gathering the required information, the branches will be merged into a comprehensive healthcare response.

        ### Inputs:
        - **Category**: {category}
        - **Original Query**: {original_query}
        - **Branch Output**: {branch_output}
        - **Response Start**: {response_start}

        ### Your Task (Solve Step):
        Based on an ambiguity score of {ambiguity_score}, your task is to generate follow-up questions for each branch based on this guideline: {question_detail_level}
        These questions should help clarify or gather additional information relevant to the category of the query (e.g., diagnosis, treatment plan, prognosis, prevention, etc.).

        - The questions should be **specific to each branch** and aligned with the nature of the healthcare query.
        - They should encourage comprehensive answers (avoid yes/no questions).
        - Ensure that the questions are **clinically relevant** and help guide further healthcare steps (e.g., diagnostic, preventive, or treatment-focused steps).

        Follow these instructions for generating the follow-up questions: {follow_up_instructions}

        ### Format:
        Return the follow-up questions as a list for each branch, formatted as follows:

        Branch 1:
        - Follow-up Question 1
        - Follow-up Question 2
        - ...

        Branch 2:
        - Follow-up Question 1
        - Follow-up Question 2
        - ...

        You **must strictly follow this format** for the response. Now, generate the **Solve** portion of the BSM workflow, focusing on the follow-up questions based on the Branch output above.
        """


        solve_output = self._planner_model.generate(query=solve_prompt).strip()
        return branch_output, solve_output


    def extract_follow_up_questions(self, response: str) -> dict:
        """
        Extracts follow-up questions from the model's response.
        
        Parameters:
            response (str): The response containing the follow-up questions.
        
        Returns:
            dict: A dictionary with branch names as keys and lists of questions as values.
        """
        # Step 1: Split the response into branches using the pattern for Branch titles (e.g., "Branch 1:", "**Branch 1**", etc.)
        branches = re.split(r'Branch \d+:|Branch \d+:\s+', response)
        
        # Step 2: Extract the branch titles (if they exist)
        branch_titles = re.findall(r'Branch \d+: ([^\n]+)', response)
        
        questions_dict = {}
        
        # Step 3: Process each branch to extract the follow-up questions
        for i, branch in enumerate(branches[1:], start=1):  # Skip the first item since it's before the first branch
            # Generate the branch name using the extracted titles if available
            if len(branch_titles) >= i:
                branch_name = f"Branch {i}: {branch_titles[i-1].strip()}"
            else:
                branch_name = f"Branch {i}"

            # Step 4: Use regex to extract both numbered questions (e.g., '1.', '2.') and bullet-point questions (e.g., '-')
            questions = re.findall(r'(?:\d+\.\s+|\-\s+)(.+)', branch.strip())
            
            # Only add the branch if there are valid questions
            if questions:
                questions_dict[branch_name] = [q.strip() for q in questions]  # Clean up extra spaces around questions
            else:
                print(f"No valid follow-up questions found for {branch_name}.")
        
        return questions_dict


    def generate_follow_up(self, category: str, tool_names: str, meta: str, previous_actions: str, history: str, input: str) -> str:
        """
        Generates a set of follow-up questions based on the category and query input.
        """
        # Generate the BSM template with the follow-up instructions
        branch_output, solve_output = self.generate_bsm_template(
            category=category,
            original_query=input,
            tool_descriptions=self.task_descriptions()
        )

        print('Branch output \n: ', branch_output)
        print('Solve output: \n', solve_output)

        #return list of follow-up questions to the user
        follow_up_questions = self.extract_follow_up_questions(solve_output)

        print("Follow-up questions before filtering:")
        count = 1
        for question_branch in follow_up_questions.values():
            for question in question_branch:
                print(f"{count}. {question}")
                count += 1

        #SOLVE with filtering all at once
        # Step 3: Concatenate all follow-up questions into a single string
        all_questions = []
        for branch, questions in follow_up_questions.items():
            all_questions.extend(questions)

        print('Tool descriptions: ', self.task_descriptions())

        # Step 4: Prepare prompt to filter the questions based on the available tools
        filter_prompt = f"""
        You are an expert clinical decision-making assistant, specializing in healthcare tool utilization. 
        Your task is to determine which of the following patient-related follow-up questions can be answered by the available tools, and which questions need to be asked directly to the patient.

        The following tools are available for gathering patient information:
        {self.task_descriptions()}

        Here are some follow-up questions that have been generated to gather information from the patient:
        {', '.join(all_questions)}

        ### Important:
        - **Do not** consider tools like `serpapi` (for web searches) and `extract_text` (for scraping text from web pages), as these do not pertain to patient data.
        - Only consider tools that deal with **patient data**, such as medical history, medications, diseases, vaccinations, and similar information.

        ### Task:
        Evaluate **each question one-by-one**. If a question can be fully answered by one of the available tools, provide the name of the tool that can answer it. 
        If a question cannot be answered by any tool, indicate that it needs to be asked directly to the patient.

        ### Format your response exactly as follows:
        1. "Question: {{question}} | Answer: yes: [Tool Name]" (if a tool can answer the question)
        2. "Question: {{question}} | Answer: no" (if the question needs to be asked to the patient)

        Note: Replace {{question}} with the actual question being evaluated.
        """

        # Step 5: Make a single API call to check all questions
        filter_output = self._planner_model.generate(query=filter_prompt).strip()
        print('Filter output: \n', filter_output)

        # Step 6: Parse the filtered output to keep only the necessary questions
        filtered_follow_up_questions = []
        lines = filter_output.split('\n')
        print(lines)
        for line in lines:
            if 'no' in line.lower():  # Keep only questions that need to be asked to the patient
                question_match = re.search(r'Question: (.+?) \|', line)
                if question_match:
                    filtered_follow_up_questions.append(question_match.group(1))

        print('Filtered follow-up questions: ', filtered_follow_up_questions)


        # Step 7: Concatenate all remaining questions into a single string for output
        concatenated_questions = "\n".join(filtered_follow_up_questions)
        
        return concatenated_questions.strip()

    
    def generate_code_prompt(self, strategy: str, tool_names: str, meta: str, previous_actions: str, history: str, input: str) -> str:
        """
        Generates a structured prompt for generating Python code based on the selected strategy.

        Args:
            strategy (str): The final strategy selected during the planning process.
            tool_names (str): The names of the tools available for use.
            meta (str): Metadata relevant to the task.
            previous_actions (str): The list of previous actions taken.
            history (str): The history of interactions.
            input (str): The user's input query.

        Returns:
            str: A structured prompt for generating Python code.
        """
        # Fill in the template with the provided arguments
        prompt = f"""
        {strategy}
        =========================
        {previous_actions}
        =========================
        Tools:
        {tool_names}
        =========================
        MetaData:
        {meta}
        =========================
        {history}
        USER: {input}

        You are a skilled Python programmer that can solve problems and convert them into Python code. 
        Using the selected final strategy mentioned in the 'Decision:', create Python code inside a ```python``` block that outlines a sequence of steps using the Tools. 
        Assume that there is a `self.execute_task` function that can execute the tools in it. The `execute_task` function 
        receives the task name and an array of the inputs and returns the result. Make sure that you always pass an array as the second argument. 
        You can call tools like this: 
        **task_result = self.execute_task('tool_name', ['input_1', 'input2', ...])**. 
        The flow should utilize this style representing the tools available. Make sure all the `execute_task` calls' outputs are stored in a variable.
        If a step's output is required as input for a subsequent step, ensure the Python code captures this dependency clearly. 
        The output variables should be directly passed as inputs with no changes in the wording.
        If the tool input is a datapipe, only put the variable as the input. 
        For each tool, include necessary parameters directly without any names and assume each will return an output. 
        The outputs' description is provided for each Tool individually. Make sure you use the directives when passing the outputs.

        Question: {input}
        """

        return prompt


    def plan(
        self,
        query: str,
        history: str = "",
        meta: str = "",
        previous_actions: List[str] = None,
        use_history: bool = False,
        **kwargs: Any,
    ) -> str:
        """
            Generate a plan using Tree of Thought

        Args:
            query (str): Input query.
            history (str): History information.
            meta (str): meta information.
            previous_actions (List[Action]): List of previous actions.
            use_history (bool): Flag indicating whether to use history.
            **kwargs (Any): Additional keyword arguments.
        Return:
            Action: return action.

        """
        # Step 1: Categorize the inquiry
        category = self.categorize_inquiry(query)

        print('Query: ', query)
        print('Category: ', category)

        #previous actions prompt
        previous_actions_prompt = ""
        if len(previous_actions) > 0 and self.use_previous_action:
            previous_actions_prompt = f"Previous Actions:\n{self.generate_scratch_pad(previous_actions, **kwargs)}"


        # Step 2: Generate a response based on the category
        history_prompt = ""
        if use_history:
            history_prompt = history
        else: 
            history_prompt = "No History"
        # prompt = self.generate_prompt(
        #     category=category,
        #     tool_names=self.task_descriptions(),
        #     meta=meta,
        #     previous_actions=previous_actions_prompt,
        #     history= history_prompt,
        #     input=query)
        

        # kwargs["max_tokens"] = 1000
        # response = self._planner_model.generate(query=prompt, **kwargs)
        # print('Response 1: ', response)

        # Generate the follow-up list based on the provided context
        follow_up_list = self.generate_follow_up(
            category=category,
            tool_names=self.task_descriptions(),
            meta=meta,
            previous_actions=previous_actions_prompt,
            history=history_prompt,
            input=query
        )

        print('Follow-up list: \n', follow_up_list)

        # Split the follow-up list into individual questions and clean up any empty lines
        follow_up_questions = [q.strip() for q in follow_up_list.split('\n') if q.strip()]

        print('Follow-up questions one by one: ', follow_up_questions)

        # Step 4: Format the entire list of follow-up questions into a single string
        formatted_follow_up = "In order to answer your query, please answer the following questions:\\n\\n"
        for question_counter, question in enumerate(follow_up_questions, 1):
            # Format each question with a number and a newline
            formatted_follow_up += f"{question_counter}. {question}\\n"

        # Escape any double quotes and newlines in the formatted string to avoid issues in the task execution
        escaped_formatted_follow_up = formatted_follow_up.replace('"', '\\"')

        # Step 5: Create a single action string to ask the user the entire list of questions at once
        askUserAction = f'self.execute_task("ask_user", ["{escaped_formatted_follow_up}"])\n'

        # Return the single action to be executed
        return askUserAction


        # kwargs["max_tokens"] = 1000
        # response = self._planner_model.generate(query=prompt, **kwargs)
        # print('Response 1: ', response)
        
        #second part of prompt - generate Python code 
        # # Assuming we have the necessary inputs
        # final_strategy = self.extract_decision_part(category, response)
        # available_tools = self.get_available_tasks()  # Get the list of available tools
        # previous_actions_prompt = previous_actions if previous_actions else "No previous actions"
        # user_query = query  # The user's original query
        # meta_data = meta  # Meta information
        # interaction_history = history if use_history else "No History"  # Use history if available

        # # Generate the Python code prompt
        # code_prompt = self.generate_code_prompt(
        #     strategy=final_strategy,
        #     tool_names=available_tools,
        #     meta=meta_data,
        #     previous_actions=previous_actions_prompt,
        #     history=interaction_history,
        #     input=user_query
        # )

        # # Now you can feed this code_prompt to the language model to generate the Python code
        # kwargs["stop"] = self._stop
        # response = self._planner_model.generate(
        #     query=code_prompt, **kwargs
        # ) #generated code
        
        # print('Response 2: ', response)

        # index = min([response.find(text) for text in self._stop])
        # response = response[0:index]
        #actions = self.parse(response)
        #return actions


    def parse(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """
            Parse the output query into a list of actions or a final answer. It parses the output based on \
            the following format:

                Action: action\n
                Action Inputs: inputs

        Args:\n
            query (str): The planner output query to extract actions.
            **kwargs (Any): Additional keyword arguments.
        Return:
            List[Union[Action, PlanFinish]]: List of parsed actions or a finishing signal.
        Raise:
            ValueError: If parsing encounters an invalid format or unexpected content.

        """
        pattern = r"`+python\n(.*?)`+"
        code = re.search(pattern, query, re.DOTALL).group(1)
        return code
    