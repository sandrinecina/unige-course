# -*- coding: utf-8 -*-
"""
Multi-tool agent using LlamaIndex (v0.10+), OpenAI, LangChain Community, and Langfuse.
- Tools: Calculator, OpenAI chat, Wikipedia
- Tracing: Langfuse @observe decorator + OpenAI native integration
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# --- Langfuse
from langfuse import get_client, observe
from langfuse.openai import openai  # auto-traces OpenAI calls

# --- LlamaIndex
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

# --- Import SDG data service
# (SDGDataService will be imported inside the tool function)

# ======================
# Setup: API Keys
# ======================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required by OpenAI + LlamaIndex wrapper

# --- OpenAI client (uses OPENAI_API_KEY from env) - moved after load_dotenv()
openai_client = openai.OpenAI()

# Set Langfuse environment variables (get_client() reads these)
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LF_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LF_PUBLIC_KEY")
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

# Initialize Langfuse client (reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
langfuse = get_client()

# ======================
# Tool: SDG API Wrapper
# ======================
# NB: With this decorator we get automatic tracing of calls to this function to Langfuse (way simpler than doing it by hand!)
@observe(name="sdg_indicators", as_type=None)
def sdg_tool(query: str) -> str:
    import json
    import re
    import os
    from langchain_core.documents import Document
    
    try:
        # Parse query to find specific SDG number
        sdg_match = re.search(r'sdg\s*(\d+)|goal\s*(\d+)', query.lower())
        requested_goal = None
        if sdg_match:
            requested_goal = sdg_match.group(1) or sdg_match.group(2)
        
        # Load SDG data from local JSON file
        json_path = os.path.join(os.path.dirname(__file__), 'sdg_indicators.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            return "SDG data file not found. Run fetch_and_save_sdg_data.py first."
        
        if not data:
            return "No SDG data available"
        
        # Process each SDG goal and extract keywords from indicators
        keyword_clusters = {}
        
        # Filter to only process the requested goal
        goals_to_process = data
        if requested_goal:
            goals_to_process = [g for g in data if g['goalId'] == requested_goal]
            if not goals_to_process:
                return f"SDG {requested_goal} not found"
        
        for goal in goals_to_process:
            goal_id = goal['goalId']
            goal_name = goal['goalName']
            
            # Create text representation for knowledge graph approach
            goal_text = f"SDG Goal {goal_id}: {goal_name}\n\n"
            goal_text += f"This Sustainable Development Goal focuses on: {goal_name}\n\n"
            goal_text += "Key indicators include:\n"
            
            # Collect all indicator descriptions with their IDs
            indicator_descriptions = []
            indicators_with_ids = []
            for indicator in goal['indicators']:
                indicator_descriptions.append(indicator['description'])
                indicators_with_ids.append({
                    'code': indicator['code'],
                    'description': indicator['description']
                })
                goal_text += f"- [{indicator['code']}] {indicator['description']}\n"
            
            # Create a Document object for LangChain
            document = Document(page_content=goal_text, metadata={"sdg_goal": goal_id})
            
            # Use OpenAI to extract keywords with knowledge graph context
            if indicator_descriptions:
                prompt = f"""
                Analyze this SDG text and extract ATOMIC SEARCH KEYWORDS for UN SDG Goal {goal_id}: {goal_name}.

                GOAL CONTEXT (MANDATORY)
                - Every output item MUST preserve goal context. If a generic term would be ambiguous, SCOPE it explicitly (e.g., "women's political participation", "gender-based violence", "gender-responsive budgeting").
                - Additionally, append the exact goal tag to each line: [sdg{goal_id} {goal_name.lower()}].

                ATOMICITY & GRANULARITY
                - ONE concept per bullet. If a phrase would require "and", "/", "+", "," or includes multiple entities/actions, SPLIT into multiple bullets.
                - 1–3 word noun phrases (4 if needed for a denominator/unit, e.g., "per 100,000 population").
                - Prefer concrete, domain-specific nouns/noun phrases: populations, harms, rights, services, assets, sectors, interventions, outcomes, infrastructures, measures, units, legal constructs.

                NORMALIZATION
                - lowercase; singular; ASCII; keep standard acronyms (hiv, gdp, ncd).
                - no stopwords; no indicator numbering.

                COVERAGE
                - If a concept appears in ANY indicator, include it ONCE.
                - Include meaningful units/denominators when intrinsic to the measure (e.g., "per 100,000 population", "percentage", "share") — but NEVER output a unit alone; it must be attached to a measure or population context (e.g., "percentage of women in parliament" is OK; "percentage" alone is NOT).

                SCOPING RULES (Goal-aware)
                - For Goal 5, add gender scope when needed: "women's", "girls'", "female", "gender-based", "gender-responsive".
                - Examples of scoping:
                - "political participation" → "women's political participation"
                - "leadership representation" → "women's leadership representation"
                - "land ownership" → "women's land ownership"
                - keep inherently gendered terms as-is: "female genital mutilation", "gender-based violence"

                BANLIST (do NOT output these ALONE)
                - Generic or meta terms: access, availability, service, system, country, location, framework, regulation, law, enforcement, monitoring, tracking, information, education (alone), age (alone), share, percentage, proportion, number, type, owner, rights-bearer, place of occurrence.
                - If such words appear, only output them when combined into a scoped, domain-specific concept (e.g., "gender-responsive budgeting", "sex-disaggregated data", "customary land tenure").

                REPHRASE RULES (fix common offenders)
                - "sex" → "sex-disaggregated data"
                - "education" (alone) → reject unless scoped (e.g., "sexual and reproductive health education")
                - "information" (alone) → reject unless scoped (e.g., "reproductive health information")
                - "system" → reject
                - "country" → reject

                OUTPUT FORMAT (STRICT)
                - Return EXACTLY a bullet list; one item per line.
                - Each line MUST start with "- " then the keyword, followed by a space and the goal tag.
                - Do NOT use "and", "&", "/", "+", or commas inside the keyword.

                NEGATIVE → POSITIVE (style mapping)
                - "- political participation" → "- women's political participation [sdg{goal_id} {goal_name.lower()}]"
                - "- leadership representation" → "- women's leadership representation [sdg{goal_id} {goal_name.lower()}]"
                - "- sex" → "- sex-disaggregated data [sdg{goal_id} {goal_name.lower()}]"
                - "- land ownership" → "- women's land ownership [sdg{goal_id} {goal_name.lower()}]"
                - "- percentage" → reject (unless scoped as part of a measure)

                CONTEXT
                Goal: {goal_name}

                INDICATORS (with IDs)
                {chr(10).join([f'[{ind["code"]}] {ind["description"]}' for ind in indicators_with_ids])}"""
                
                # Use the openai_chat function to extract keywords
                keywords_response = openai_chat(prompt)
                keyword_clusters[f"Goal {goal_id}"] = {
                    "name": goal_name,
                    "keywords": keywords_response,
                    "indicator_count": len(indicator_descriptions),
                    "indicators": indicators_with_ids
                }
        
        # Format the response
        response = "SDG Keyword Clusters:\n\n"
        for goal_key, cluster_data in keyword_clusters.items():
            response += f"{goal_key}: {cluster_data['name']}\n"
            response += f"Keywords: {cluster_data['keywords']}\n"
            response += f"(Based on {cluster_data['indicator_count']} indicators)\n\n"
        
        # Save to output file
        from datetime import datetime
        output_filename = f"sdg_keywords_output_{requested_goal if requested_goal else 'all'}.txt"
        output_path = os.path.join(os.path.dirname(__file__), output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"SDG Keywords Extraction\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 50}\n\n")
            
            for goal_key, cluster_data in keyword_clusters.items():
                f.write(f"{goal_key}: {cluster_data['name']}\n")
                f.write(f"Based on {cluster_data['indicator_count']} indicators\n")
                f.write(f"{'-' * 40}\n")
                
                # Write indicators with their IDs
                f.write("Indicators analyzed:\n")
                for ind in cluster_data['indicators']:
                    f.write(f"  [{ind['code']}] {ind['description']}\n")
                f.write(f"\n{'-' * 40}\n")
                
                # Write keywords
                f.write("Extracted keywords:\n")
                f.write(f"{cluster_data['keywords']}\n")
                f.write(f"\n{'=' * 50}\n\n")
        
        response += f"\n📁 Output saved to: {output_filename}"
        
        # Also save as JSON for programmatic use
        json_filename = f"sdg_keywords_output_{requested_goal if requested_goal else 'all'}.json"
        json_path = os.path.join(os.path.dirname(__file__), json_filename)
        
        # Prepare clean JSON data
        json_data = {}
        for goal_key, cluster_data in keyword_clusters.items():
            # Parse keywords from the response string
            keywords_list = []
            for line in cluster_data['keywords'].strip().split('\n'):
                if line.startswith('- '):
                    # Remove the bullet and the [sdg... tag]
                    keyword = line[2:].split('[sdg')[0].strip()
                    if keyword:
                        keywords_list.append(keyword)
            
            json_data[goal_key] = {
                "name": cluster_data['name'],
                "indicator_count": cluster_data['indicator_count'],
                "indicators": cluster_data['indicators'],
                "keywords": keywords_list
            }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        response += f"\n📁 JSON output saved to: {json_filename}"
        
        return response
        
    except Exception as e:
        return f"Error processing SDG data: {str(e)}"

sdg_func_tool = FunctionTool.from_defaults(
    fn=sdg_tool,
    name="SDG_Keyword_Extractor",
    description="Extract and cluster keywords from all SDG indicators using AI analysis. Returns keyword clusters for each SDG goal."
)


# ======================
# Tool: OpenAI Chat --> We see here that we can pass any LLM inside our agent, as a tool (e.g. we could pass a Claude tool inside an OpenAI agent)
# ======================
@observe(name="openai-chat", as_type="generation")
def openai_chat(prompt: str) -> str:
    """
    Uses OpenAI Chat Completions API (gpt-4.1 family).
    Langfuse traces automatically via langfuse.openai.
    """
    resp = openai_client.chat.completions.create(
        model="gpt-4.1",  # or "gpt-4.1-mini"
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

openai_tool = FunctionTool.from_defaults(
    fn=openai_chat,
    name="OpenAIChat",
    description="Ask a general knowledge question via OpenAI."
)


# ======================
# Agent Setup (LlamaIndex)
# ======================
TOOLS = [sdg_func_tool]
llm = LlamaIndexOpenAI(model="gpt-4.1", api_key=OPENAI_API_KEY)
agent = FunctionAgent(
    tools=TOOLS,
    llm=llm,
    system_prompt="""
    Extract ATOMIC, goal-scoped SDG keywords.

INPUT
- goal_id: string or int (e.g., "5")
- goal_name: string (e.g., "Gender Equality")
- indicator_descriptions: array of strings (raw indicator texts)

RULES
- One concept per item (atomic).
- Enforce goal context in each item (e.g., prefix/scope or tag).
- Normalize: lowercase, singular, ascii; allow acronyms (hiv, gdp).
- Deduplicate.
- No coordinators ("and", "&", "/", "+", commas) inside items.

OUTPUT (JSON)
{
  "keywords": [
    {
      "term": "women's political participation",     // atomic & scoped
      "goal_id": "5",
      "goal_name": "Gender Equality",
      "indicator_id": "5.5.1",                    
      "tags": ["sdg5","gender-equality"]            // include a short slug
    },
    ...
  ]
}
    # You are a thematic keyword analyzer. Your role is to use the tool at your disposal to cluster Do not answer directly, only use the provided tools. If you don't know the answer from the tools, just say you don't know.
    """
    # verbose=True
)

# ======================
# Async CLI app
# ======================
async def main():
    # On Windows, ensure a selector loop policy
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("LlamaIndex OpenAI Agent (exit with 'exit' or 'quit')")
    loop = asyncio.get_running_loop()

    while True:
        # non-blocking input in async code
        query = await loop.run_in_executor(None, input, "\n> ")
        if query.strip().lower() in ("exit", "quit"):
            break

        # Group each turn under a Langfuse span (now there IS a running loop)
        with langfuse.start_as_current_span(name="user-turn", metadata={"source": "cli"}):
            res = await agent.run(user_msg=query)
            print("\n" + str(res))
            langfuse.update_current_trace(tags=["cli", "demo"])

        # Flush Langfuse buffers
        langfuse.flush()
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())