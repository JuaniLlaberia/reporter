from langchain_core.prompts import ChatPromptTemplate

GENERATE_PLAN_PROMPT = ChatPromptTemplate.from_template("""
You are a report planning expert. Your task is to generate a set of sections for the report.

Each section will contain (All this fields must be present):
- name: A short, descriptive title for the section.
- expected_format: A list of one or more formats from ["narrative", "bullets", "table"].
- goal: A clear, concise statement describing the purpose of the section — what insight or message it should deliver to the reader.
- queries: Exactly 5 well-phrased, diverse search queries to retrieve information from a vector database for this section.
- img_query: 1 well-phrased query to retrieve an image (searching its description) from a vector database for this section.

Guidelines for queries:
- Rephrase and vary them to avoid duplicates.
- Include key entities, metrics, or constraints where relevant.
- Cover different perspectives for completeness.

Guidelines for image query:
- Specify visual attributes.
- Clearly state what the image should convey. Is it meant to "visualize growth," "represent a strategic concept," or "show a product's interface"?
- Reference key entities and actions: Name the main subjects and the actions they are performing (e.g., scaling, connecting, analyzing).
- Avoid generic terms: Be as specific as possible. Instead of "chart," use "bar chart" or "pie chart." Instead of "strategy," use "conceptual roadmap" or "strategic pillars."

Report context:
- Topic: {topic}
- Focus keys: {focus_keys}
- Report type: {type}
- General plan chunks: {plan_chunks}

Improvement notes (optional):
If `improvements` is provided, integrate them into the new plan:
{improvements}
""")


EVALUATE_PLAN_PROMPT = ChatPromptTemplate.from_template("""
You are a report analysis and evaluation expert.
Your task is to assess the quality of the following report plan and its sections, then provide a score and improvement suggestions.

Scoring criteria (0 to 1 scale):
1. Relevance: Sections align with the given topic, focus keys, and report type.
2. Coverage: The plan addresses all major aspects of the topic without significant omissions or redundancy.
3. Clarity: Section titles and goals are concise, specific, and easy to understand.
4. Query quality: Queries are diverse, well-phrased, and useful for retrieving relevant information from a vector database.

Instructions:
- Provide a 'score' between 0 (very poor) and 1 (perfect).
- If applicable, provide a list of 'improvements' to make the plan better.
- If the plan is already optimal, return an empty list for 'improvements'.

Planned sections:
{sections}
""")
