from langchain_core.prompts import ChatPromptTemplate

PROCESS_PROMPT_PROMPT = ChatPromptTemplate.from_template("""
You are a prompt analyzer expert. Your task is to extract the following information from the user prompt.
The user prompt aims to create a report so we need to extract the key aspects.

Information to extract:
- topic: The main topic of the report.
- focus_keys: The list of sub-topics that we want to report to be focus on.
- type: The report type following the enum.
- plan_queries: Provide 10 queries to retrieve information from a vectorDB to have the main idea of what we need.

Prompt: {prompt}
""")