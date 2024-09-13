"""
Heavily borrowed from langchain: https://github.com/langchain-ai/langchain/
"""
import re
from typing import Any
from typing import List

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
    
    def categorize_inquiry(self, inquiry: str) -> str:
        prompt = f"""
        Please categorize the following medical question into one of the following categories:
        - Differential Diagnosis
        - Treatment Options
        - Preventive Measures
        - Prognosis
        
        Question: {inquiry}

        Response Format:
        {{
            "category": "Differential Diagnosis"
        }}
        """
        response = self._planner_model.generate(query=prompt)

        # Parse the category from the response
        category_match = re.search(r'"category":\s*"([^"]+)"', response)
        return category_match.group(1).lower().strip() if category_match else "uncategorized"

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

    def generate_bsm_template(self, category: str, instructions: str, response_start: str, original_query: str) -> str:
        # Step 1: Generate the Branch portion based on the category
        branch_prompt = f"""
        You are a skilled health assistant tasked with restructuring an original prompt into a Branch-Solve-Merge workflow.

        ### Inputs:
        - **Category**: {category}
        - **Original Query**: {original_query}
        - **Original Instructions**: {instructions}
        - **Response Start**: {response_start}

        ### Task - Branch:

        Start by dividing the task into two or more distinct paths based on the patient’s symptoms and medical history provided in the original query. 
        Each path should represent a potential diagnosis or treatment option grounded in the provided symptoms.

        For each path, outline the main areas where follow-up questions will be necessary to gather more detailed information from the patient. 
        Focus on the high-level structure of what needs to be clarified in each path.

        Now, generate the Branch portion of the BSM workflow based on the category "{category}". This will guide the follow-up questions in the Solve phase.
        """

        branch_output = self._planner_model.generate(query=branch_prompt).strip()

        # Step 2: Generate the Solve portion
        solve_prompt = f"""
        You are continuing the task of restructuring an original prompt into a Branch-Solve-Merge workflow.

        ### Original Query:
        {original_query}

        ### Branch Output:
        {branch_output}

        ### Task - Solve:

        For each path generated in the Branch step, focus exclusively on generating a list of **follow-up questions** to ask the user. 
        These questions should be tailored to the specific branch and should help clarify or gather additional necessary information relevant to the branch’s focus (e.g., diagnosis, treatment plan, prognosis, or prevention). 

        Make sure the follow-up questions are:
        1. **Detailed and open-ended** – Encourage comprehensive responses from the user, avoiding simple yes/no questions.
        2. **Specific to each branch** – Each question should align with the hypothesis or diagnosis of the respective branch.
        3. **Clinically relevant** – Ensure that the questions are meaningful in gathering information to guide further diagnostic steps or treatment options.

        Return the follow-up questions as a list for each branch, formatted as follows:

        Branch 1:
        - Follow-up Question 1
        - Follow-up Question 2
        - ...

        Branch 2:
        - Follow-up Question 1
        - Follow-up Question 2
        - ...

        You **must strictly follow this format** for the response. Now, generate the Solve portion of the BSM workflow, focusing only on the follow-up questions based on the Branch output above.
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
        branches = re.split(r'Branch \d+:', response)
        questions_dict = {}
        
        for i, branch in enumerate(branches[1:], start=1):
            branch_name = f"Branch {i}"
            questions = re.findall(r"- (.+)", branch.strip())
            questions_dict[branch_name] = questions
    
        return questions_dict
    def extract_decision_part(self, category: str, response: str) -> str:
        # Map categories to their corresponding response starts
        response_starts = {
            "differential diagnosis": "Final diagnosis:",
            "treatment options": "Treatment options:",
            "preventive measures": "Preventive measures:",
            "prognosis": "Prognosis:"
        }

        # Get the response start phrase for the category
        response_start = response_starts.get(category)

        # Extract the decision part if the response start is found in the response
        decision_part = response.split(response_start)[-1].strip() if response_start in response else "No decision part found."

        # Return the formatted final strategy
        return f"Decision:\n{decision_part}"

    def generate_follow_up(self, category: str, tool_names: str, meta: str, previous_actions: str, history: str, input: str) -> str:
        category_mapping = {
            "differential diagnosis": {
                "instructions": """Use the tools and provided information to first suggest two or more possible diagnoses \
                    for the patient based on their symptoms and medical history. \
                    Give a detailed explanation consisting of sequences of tools to properly come up with the diagnoses to answer the user query. \
                    For each diagnosis, list the steps, tests, or treatments you would recommend to confirm or address it. \
                    Finally, combine the results from all the diagnoses into a comprehensive treatment plan or next steps.""",
                "response_start": "Final diagnosis:"
            },
            "treatment options": {
                "instructions": """Use the tools and provided information to generate detailed treatment options for the patient. This should include a list of possible treatments, including medications, therapies, and lifestyle changes. For each treatment, include potential side effects and success rates based on the patient's medical history.""",
                "response_start": "Treatment options:"
            },
            "preventive measures": {
                "instructions": """Use the tools and provided information to suggest preventive measures for the patient based on their current health condition and medical history. Generate a list of preventive measures that can help the patient avoid future health issues. Ensure that the suggestions are practical and actionable.""",
                "response_start": "Preventive measures:"
            },
            "prognosis": {
                "instructions": """Use the tools and provided information to provide a prognosis for the patient. This should include an assessment of the likely course and outcome of the patient's condition. Ensure that the information is accurate and clear.""",
                "response_start": "Prognosis:"
            }
        }

        # Get the category-specific instructions and response start
        instructions = category_mapping[category]["instructions"]
        response_start = category_mapping[category]["response_start"]

        # Generate the BSM template
        branch_output, solve_output = self.generate_bsm_template(category, instructions, response_start, input)
        print('Branch output \n: ', branch_output)
        print('Solve output: \n', solve_output)

        #return list of follow-up questions to the user
        follow_up_questions = self.extract_follow_up_questions(solve_output)

        print('Follow up questions before filtering: \n', follow_up_questions)

        # Step 5: Get available tools and descriptions
        tool_descriptions = tool_names  # Use self.task_descriptions() to get the tools and their descriptions

        print('tool descriptions: \n', tool_descriptions)

        # Step 6: Modify helper function to use ChatGPT (or language model) for dynamic decision making
        def is_tool_available_for_question(question: str) -> bool:
            """
            This helper function asks ChatGPT whether a tool can answer the given question based on the tool descriptions.
            Returns a bool indicating if a tool is available and prints the tools if applicable.
            
            Args:
            - question (str): The follow-up question for the patient.
            - tool_descriptions (str): A formatted string that lists and describes available tools.
            - available_tasks (List[str]): A list of available task names that the model can choose from.
            
            Returns:
            - bool: True if one or more tools are available, False otherwise.
            """
            prompt = f"""
            The following tools are available:
            {tool_descriptions}

            The task is to assess the patient's condition by asking relevant follow-up questions. These questions are aimed at understanding the patient's symptoms and health status, not extracting text from web pages or performing tasks unrelated to healthcare.

            Can any of these tools (excluding `ask_user`) help answer the following follow-up question for a patient?
            
            Follow-up question: "{question}"

            Please evaluate whether any of the tools can answer the question based on their **relevance to gathering health information from the patient**. 
            Do not respond "yes" if the only tool that can answer the question is `ask_user`. This task is currently being used to ask follow-up questions that other tools cannot handle. 
            Tools that are used for scraping text from web pages (such as `extract_text`) are irrelevant to this context.

            Respond with "yes" or "no". If "yes", explicitly list the relevant tool(s) by name. 
            You **must strictly follow this format** for the response:
            - "yes: [Tool Name 1], [Tool Name 2], ..."
            - "no"
            
            Avoid suggesting tools unrelated to understanding the patient's condition.
            """
            # Use the language model to evaluate the question against the tool descriptions
            response = self._planner_model.generate(query=prompt).lower().strip()

            print('is tool available: ', response)

            # If the model responds with "yes" and mentions any tool, return True
            if "yes" in response:
                return True
            return False

        # Step 7: Filter out questions that can be answered by available tools
        filtered_follow_up_questions = {}
        number_skipped = 0
        for branch, questions in follow_up_questions.items():
            filtered_questions = []
            for question in questions:
                print('question: ', question)
                if not is_tool_available_for_question(question):
                    filtered_questions.append(question)
                else:
                    number_skipped += 1
                    print(f"Skipping question: '{question}' as it can be handled by an available tool.")
            if filtered_questions:
                filtered_follow_up_questions[branch] = filtered_questions
        print('number skipped: ', number_skipped)

        # Step 2: Concatenate all questions into a single string
        concatenated_questions = ""
        for questions in filtered_follow_up_questions.values():
            concatenated_questions += " \n".join(questions) + " "

        return concatenated_questions.strip()


        # # Fill in the template
        # prompt = f"""
        # As a knowledgeable and empathetic health assistant, your primary objective is to provide the user with precise and valuable information regarding their health and well-being. Utilize the available tools effectively to answer health-related queries. Here are the tools at your disposal:
        # {tool_names}

        # The following is the format of the information provided:
        # MetaData: this contains the name of data files of different types like image, audio, video, and text. You can pass these files to tools when needed.

        # Follow this plan: {bsm_prompt_template}

        # Based on generated plan, write the detailed tool executions step by step. Ensure that your final response is comprehensive and uses the proper tools. **You should stick to the provided tools and never use any other tasks.** Never try to load files as your strategy steps.

        # Start your final response with
        # '{response_start}'.

        # Begin!

        # MetaData:
        # {meta}
        # =========================
        # {previous_actions}
        # =========================
        # {history}
        # USER: {input}
        # CHA:
        # """

        # return prompt
    
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
    