import json
import logging
import streamlit as st
from retry import retry
from langchain.chains import GraphCypherQAChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama

# ✅ Updated Cypher generation prompt template
CYPHER_GENERATION_TEMPLATE = """Task: You are a Cypher expert. Generate an accurate Cypher query ONLY using the schema below.

Instructions:
1. Use only the node labels, relationship types, and properties from the schema.
   Avoid adding labels like `:Location` unless explicitly required. Prefer using just `{{Name: "..."}}` if the label is not critical.
2. Return fields like paper.title, location.name — never return paths like `p`.
3. Always do case-insensitive partial string matching:
   Use `toLower()` on both sides like `toLower(node.property) CONTAINS toLower("value")` to make matching safe.
4. Always wrap the Cypher query in triple backticks (```)
5. Add LIMIT 10 unless specified otherwise.

Schema:
{schema}

Example Question 1: Which papers mention anomalous temperature regimes such as cold air outbreaks (CAOs) or warm waves (WWs) in relation to North America, specifically in the sentences where these terms appear?

Example Cypher:
MATCH (we)-[:TargetsLocation]-(l{{Name:"NORTH_AMERICA"}})
MATCH (p:Paper)-[m:Mention]-(we) 
WHERE toLower(m.Mention_Sentence) CONTAINS toLower("WW") OR toLower(m.Mention_Sentence) CONTAINS toLower("CAOs")
RETURN p, l, we;

Now generate a Cypher query for:

{question}
"""



CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

MEMORY = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key='question', 
    output_key='answer', 
    return_messages=True
)

# Neo4j connection
url = st.secrets["NEO4J_URI"]
username = st.secrets["NEO4J_USERNAME"]
password = st.secrets["NEO4J_PASSWORD"]

graph = Neo4jGraph(
    url=url,
    username=username,
    password=password,
    sanitize=True
)

# ✅ Custom schema manually defined for prompt clarity
custom_schema = """
Nodes:
- Paper(title: String)
- WeatherEvent(Name: String)
- Model(Name: String)
- Teleconnection(Name: String)
- OceanCirculation(Name: String)
- Location(Name: String)
- Project(Name: String)

Relationships:
- (Paper)-[:Mention {Mention_Sentence: String}]->(WeatherEvent)
- (Paper)-[:Mention {Mention_Sentence: String}]->(Model)
- (Paper)-[:Mention {Mention_Sentence: String}]->(Teleconnection)
- (Paper)-[:Mention {Mention_Sentence: String}]->(OceanCirculation)
- (WeatherEvent)-[:TargetsLocation]->(Location)
- (Model)-[:TargetsLocation]->(Location)
- (Teleconnection)-[:TargetsLocation]->(Location)
- (OceanCirculation)-[:TargetsLocation]->(Location)
"""

# LangChain chain with Ollama model
graph_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOllama(model="qwen2", temperature=0),
    qa_llm=ChatOllama(model="qwen2", temperature=0),
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,  # ✅ pass the custom prompt
    validate_cypher=True,
    return_direct=True,
    verbose=True,
    allow_dangerous_requests=True
)


@retry(tries=2, delay=12)
def get_results(question) -> str:
    """Generate a response from the GraphCypherQAChain using a cleaned schema and improved prompt."""
    
    logging.info(f'Using Neo4j database at URL: {url}')
    graph.refresh_schema()

    # 🔙 Log full Neo4j schema in terminal
    print("\n========= Raw Schema from Neo4j =========\n")
    print(graph.get_schema)

    # ✅ Use custom schema instead of auto-generated
    prompt = CYPHER_GENERATION_PROMPT.format(schema=custom_schema, question=question)
    print('\n========= Prompt to LLM =========\n')
    print(prompt)

    try:
        chain_result = graph_chain.invoke(
            {"query": question},
            prompt=prompt,
            return_only_outputs=True,
        )
    except Exception as e:
        logging.warning(f'Handled exception running GraphCypher chain: {e}')
        return "Sorry, I couldn't find an answer to your question"

    if chain_result is None:
        return "No answer was generated."

    # ✅ Debug: show Cypher used
    cypher_query = chain_result.get("cypher", "No Cypher returned")
    print("\n========= Generated Cypher Query =========\n")
    print(cypher_query)

    result = chain_result.get("result", None)
    print("\n========= Final Result =========\n")
    print(json.dumps(chain_result, indent=2))

    return result
