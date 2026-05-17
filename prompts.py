from typing import Literal




########### ProcessBench prompts ###########

processbench_earlystep_onepass="""
The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}.
""".strip()


processbench_earlystep_onepass_careful_v1="""
The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Review each paragraph carefully for mathematical correctness and logical consistency. Be as critical as possible.
When you analyze each paragraph, you should use proper verification, recalculation, or reflection to indicate whether it is logically and mathematically valid. Please elaborate on the analysis process carefully.

Please put your final answer (i.e., the index) in \\boxed{{}}.
""".strip()


h2v_error_id_system_prompt = """
You are a strict, reliable math grader performing an Error identification task.
Your task is to identify the first incorrect paragraph in a mathematical solution.
Return your answer using the exact format requested by the user.
""".strip()

h2v_error_id_user_prompt = (
    "The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Solution]\n\n"
    "{tagged_response}\n\n"
""" 
Your task is to identify the first incorrect paragraph in the solution.

Instructions:
- Review each paragraph carefully for mathematical correctness and logical consistency
- A paragraph is incorrect if it contains mathematical errors, logical inconsistencies, or is based on errors from previous paragraphs
- Find the FIRST paragraph that is incorrect (0-indexed)
- If ALL paragraphs are correct, return -1

Provide your reasoning and then put your final answer (i.e., the index) in \\boxed{{}}.
""".strip()
)



PROMPT_PROCESS_SYSTEM_ERROR_ID = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user prompt displayed below. You will be given the assistant's solution to a math problem, which is split into steps, starting with a <step [step number]> tag, where [step number] is indexed from 0. Your job is to identify which step an error occurs, if an error is present.
When evaluating the solution, consider each step separately. Evaluate the content of each step for correctness. If you encounter a mistake at <step [step number]>, output [step number] as your Verdict. If the full response is error free, then select step number -1. Avoid any biases, such as length of step, or stylistic elements like formatting.

Here are some rules for evaluation.
(1) The assistant's answer does not need to be complete or arrive at a final solution. You may receive a partially complete response. Your job is to assess the quality of each step.
(2) When evaluating the assistant's answer, identify any mistakes or inaccurate information. Focus on the content each step and determine if the step is logically valid.
(3) For each step, you should provide an explanation of your assessment. If you find an error, describe the nature and cause of the error.
(4) Avoid any biases, such as answer length, or stylistic elements like formatting.

Before providing an your final verdict, think through the judging process and output your thoughts as an explanation
After providing your explanation, you must output the corresponding step number with an error. Use the following format:
Explanation: Your explanation here
Verdict: The step number with the error or -1 if no error occurs
""".strip()




PROMPT_SINGLE="""
[User Question]
{instruction}

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]
""".strip()



