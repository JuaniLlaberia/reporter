from langchain_core.prompts import ChatPromptTemplate

GENERATE_SECTION_CONTENT_PROMPT = ChatPromptTemplate.from_template("""
You are a section content writer expert for reports.

Your task:
Generate the content for this section based on the provided plan, documents, context, and optional improvement notes.

Guidelines:
- Use only the formats listed in `expected_format` for each ContentItem's `content_type`.
- Ensure the generated content aligns with the `goal` and makes effective use of the provided `documents`.
- Organize the content into 2-5 ContentItems, each with a unique `order` starting from 1.
- For `narrative`, fill the `text` field with a well-written paragraphs (enough to explain everything).
- For `bullets`, use the `items` list for bullet points (leave `text` empty).
- For `table`, provide `headers` and `rows` (leave `text` and `items` empty).
- Keep tone and style consistent and professional.

Planned content and context for this section:
- Name: {name}
- Goal: {goal}
- Documents: {documents}
- Expected formats of content: {expected_format}

Improvement notes (optional):
If `improvements` is provided, integrate them into the section content:
{improvements}
""")

EVALUATE_CONTENT_PROMPT = ChatPromptTemplate.from_template("""
You are a section content analysis and evaluation expert.
Your task is to assess the quality of the following section content and its parts, then provide a score and improvement suggestions.

Scoring criteria (0 to 1 scale):
1. Accuracy & Relevance: Content matches the planned goal and uses information from the provided documents.
2. Completeness: All important points relevant to the goal are covered.
3. Clarity & Structure: Content is well-organized, easy to read, and matches the expected formats.
4. Consistency: Tone, terminology, and style are coherent.

Instructions:
- Provide a 'score' between 0 (very poor) and 1 (perfect).
- If applicable, provide a list of 'improvements' to make the section content better.
- If the section content is already optimal, return an empty list for 'improvements'.

Section content to evaluate:
{section_content}

Planned content for this section (context):
- Name: {name}
- Goal: {goal}
- Documents: {documents}
- Expected formats of content: {expected_format}
""")