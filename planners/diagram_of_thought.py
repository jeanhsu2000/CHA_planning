import re
from typing import Any, List
from planners.action import Action
from planners.action import PlanFinish
from planners.planner import BasePlanner
from tasks.task import OutputType
from tasks.internals.ask_user import AskUser

class DiagramOfThoughtPlanner(BasePlanner):
    """
    **Description:**

        This class merges the Branch-Solve-Merge (BSM) and Diagram of Thought (DoT) planning approaches. It employs dynamic problem decomposition and solution synthesis strategies 
        while following a structured reasoning flow:
        
        1. **Proposer**: Propose multiple reasoning steps or solution pathways (branches).
        2. **Critic**: Critically evaluate each branch for accuracy, logical soundness, and feasibility.
        3. **Summarizer**: Synthesize valid reasoning steps from each branch into a final, merged solution.

        `BSM Paper <https://arxiv.org/abs/2310.15123>`_
    """

    summarize_prompt: bool = True
    max_tokens_allowed: int = 10000

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @property
    def _planner_type(self):
        return "merged-branch-solve-diagram-of-thought-planner"

    @property
    def _planner_model(self):
        return self.llm_model
    
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


    def propose_branches(self, problem_statement: str) -> List[str]:
        """
        **Proposer Role**: Generates multiple solution pathways (branches) for the given problem.
        Returns:
            List of proposed branches (distinct reasoning steps).
        """
        proposer_prompt = f"""
        <proposer>
        Objective: Propose multiple potential solutions or reasoning steps (branches) for the following problem.
        Problem: {problem_statement}

        Instructions:
        - Generate at least two distinct branches that explore different solution pathways.
        - Be concise but ensure the reasoning is clear.
        - Output Format: Enclose each solution within <branch> tags.
        </proposer>
        """
        response = self._planner_model.generate(query=proposer_prompt)
        branches = re.findall(r"<branch>(.*?)</branch>", response, re.DOTALL)
        return [branch.strip() for branch in branches]

    def critique_branches(self, branches: List[str]) -> List[str]:
        """
        **Critic Role**: Critically evaluates each proposed branch for accuracy, consistency, and logical soundness.
        Returns:
            List of critiques corresponding to each branch.
        """
        critiques = []
        for branch in branches:
            critic_prompt = f"""
            <critic>
            Objective: Critically evaluate the following proposed branch for logical consistency, accuracy, and feasibility.
            Branch: {branch}

            Instructions:
            - Highlight any errors or areas for improvement.
            - Suggest refinements where applicable.
            - Output Format: Enclose your critique within <critic> tags.
            </critic>
            """
            response = self._planner_model.generate(query=critic_prompt)
            critiques.append(re.search(r"<critic>(.*?)</critic>", response, re.DOTALL).group(1).strip())
        return critiques

    def summarize_branches(self, branches: List[str], critiques: List[str]) -> str:
        """
        **Summarizer Role**: Synthesizes valid branches and critiques into a final solution.
        Returns:
            Final merged solution based on the branches and critiques.
        """
        summarizer_prompt = f"""
        <summarizer>
        Objective: Synthesize the following branches and critiques into a final solution.
        Branches: {branches}
        Critiques: {critiques}

        Instructions:
        - Combine valid reasoning steps from each branch into a cohesive solution.
        - Ensure that the final solution addresses any issues raised by the critic.
        - Output Format: Enclose your summary within <summarizer> tags.
        </summarizer>
        """
        response = self._planner_model.generate(query=summarizer_prompt)
        return re.search(r"<summarizer>(.*?)</summarizer>", response, re.DOTALL).group(1).strip()

    def generate_follow_up(self, solution: str, available_tools: str) -> str:
        """
        Generate follow-up questions based on the solution and evaluate if any tools can help answer them.
        Returns:
            A list of follow-up questions with tool applicability.
        """
        follow_up_prompt = f"""
        You are tasked with generating follow-up questions to clarify the patient's condition based on the following solution:
        - Solution: {solution}

        Instructions:
        - Generate detailed and open-ended follow-up questions.
        - After generating the questions, evaluate whether any of the following tools can help answer them.

        Available Tools:
        {available_tools}

        Return the list of follow-up questions along with an evaluation of whether a tool can answer each question.
        """
        response = self._planner_model.generate(query=follow_up_prompt)
        return response.strip()

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
        Execute the full **Merged Branch-Solve-Diagram-of-Thought** planning cycle:
        
        1. **Propose**: Generate multiple solution branches.
        2. **Critique**: Critique the proposed branches.
        3. **Summarize**: Merge valid solutions from the branches.
        4. **Generate Follow-up Questions**: Identify any unresolved questions and evaluate tools for answering them.
        """
        # Step 1: Propose multiple solution branches
        proposed_branches = self.propose_branches(query)
        print(f"Proposed Branches: {proposed_branches}")

        # Step 2: Critique each branch
        critiques = self.critique_branches(proposed_branches)
        print(f"Critiques: {critiques}")

        # Step 3: Summarize and merge valid branches into a final solution
        final_solution = self.summarize_branches(proposed_branches, critiques)
        print(f"Final Merged Solution: {final_solution}")

        # Step 4: Generate follow-up questions based on the solution
        available_tools = self.task_descriptions()  # Get tool descriptions
        follow_up_response = self.generate_follow_up(final_solution, available_tools)
        print(f"Follow-up Questions: {follow_up_response}")

        return follow_up_response

    def parse(self, query: str) -> str:
        """
        Parse the output into structured reasoning steps or final actions based on Diagram-of-Thought roles.
        Returns:
            A dictionary of parsed components (proposer steps, critic responses, summarizer output).
        """
        proposer_steps = re.findall(r"<proposer>(.*?)</proposer>", query, re.DOTALL)
        critic_responses = re.findall(r"<critic>(.*?)</critic>", query, re.DOTALL)
        summarizer_output = re.search(r"<summarizer>(.*?)</summarizer>", query, re.DOTALL)

        return {
            "proposer_steps": proposer_steps,
            "critic_responses": critic_responses,
            "summarizer_output": summarizer_output.group(1) if summarizer_output else ""
        }
