from langchain_core.prompts import ChatPromptTemplate

GENERATE_SECTION_CONTENT_PROMPT = ChatPromptTemplate.from_template("""
You are a specialized AI assistant for generating professional report sections. Your goal is to create content that is clear, accurate, and directly addresses the section's purpose.

Task:
Generate a list of 2-4 **ContentItem** objects for a report section. Each object must have a unique `order` starting from 1.
The content must be generated based on the provided section plan, documents, and context.

ContentItem Types and Fields:
You will only use one of the following `content_type` formats for each item: `narrative`, `bullets`, or `table`.

1.`narrative`: Use this for well-structured paragraphs that explain complex topics or provide detailed analysis.
  - Required fields: `content_type` (set to `narrative`), `text`.
  - Empty fields: `items_subtitle`, `items`, `headers`, `rows`, `table_subtitle`, `table_footer`.

2.`bullets`: Use this for concise, scannable lists of key points.
  - Required fields: `content_type` (set to `bullets`), `items_subtitle`, `items`.
  - Empty fields: `text`, `headers`, `rows`, `table_subtitle`, `table_footer`.

3.`table`: Use this for presenting structured data.
  - Required fields: `content_type` (set to `table`), `headers`, `rows`, `table_subtitle`, `table_footer`.
  - Empty fields: `text`, `items_subtitle`, `items`.

Section Context:
- Name: {name}
- Goal: {goal}
- Documents: {documents}
- Expected Formats: {expected_format}

Instructions:
1. Prioritize Clarity and Accuracy: Ensure the content is professional, factually correct, and directly tied to the section's `goal` and the provided `documents`.
2. Strict Adherence to Format: Only use the `content_type` formats specified in `expected_format`. Fill only the required fields for the chosen format and leave all other fields empty.
3. No Markdown or Placeholders: Do not use markdown syntax (e.g., `#`, `*`, `_`) within the generated text, and do not use placeholders like "[Company Name]".

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