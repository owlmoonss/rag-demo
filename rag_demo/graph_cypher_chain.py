import json
import logging
import streamlit as st
from retry import retry
from langchain.chains import GraphCypherQAChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama

CYPHER_GENERATION_TEMPLATE = """
Task: Generate an accurate Cypher query ONLY using the schema below.


Schema:
{schema}

This example:

Question:
Which papers mention anomalous temperature regimes such as cold air outbreaks (CAOs) or
warm waves (WWs) in relation to North America, specifically in the sentences where these
terms appear?
Answer:
MATCH (we)-[:TargetsLocation]-(l{{Name:"NORTH_AMERICA"}})
MATCH (p:Paper)-[m:Mention]-(we)
WHERE toLower(m.Mention_Sentence) CONTAINS toLower("WW") 
   OR toLower(m.Mention_Sentence) CONTAINS toLower("CAOs")
RETURN p, l, we;


Question:
Which papers discuss ocean circulation processes—such as thermohaline circulation—in oceanic
regions that include either “North” or “South” in their names?
Answer:
MATCH (n:Location) 
WHERE toLower(n.Name) CONTAINS toLower("OCEAN") 
  AND (toLower(n.Name) CONTAINS toLower("NORTH") OR toLower(n.Name) CONTAINS toLower("SOUTH"))
MATCH (oc:OceanCirculation)-[:TargetsLocation]-(l)
MATCH (p:Paper)-[m:Mention]-(oc)
WHERE toLower(m.Mention_Sentence) CONTAINS toLower("thermohaline circulation")
RETURN n, oc, p;


Question:
Which papers mention CMIP5 models and the North Atlantic Oscillation (NAO) in the context of
the Southeast United States?
Answer:
MATCH (p:Paper)-[r:Mention]->(m:Model|Project)
WHERE toLower(m.Name) CONTAINS toLower("CMIP_5")
MATCH (p)-[t:Mention]-(n:Teleconnection{{Name:"NORTH_ATLANTIC_OSCILLATION"}})
WHERE toLower(t.Mention_Sentence) CONTAINS toLower("Southeast")
RETURN p, m, n;


Question:
Which papers mention the Pacific-North American (PNA) pattern in connection with locations in
the United States?
Answer
MATCH (p:Paper)-[:Mention]->(t:Teleconnection{{Name:"PACIFIC_NORTH_AMERICAN_PNA_PATTERN"}})
MATCH (t)-[:TargetsLocation]-(l:Location)
MATCH (p)-[:Mention]-(l)
WHERE toLower(l.wikidata_description) CONTAINS toLower("United States")
RETURN p, t, l;



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


graph_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOllama(model="qwen2", temperature=0),
    qa_llm=ChatOllama(model="qwen2", temperature=0),
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,  
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
    #print("\n========= Raw Schema from Neo4j =========\n")
    #print(graph.get_schema)

    # ✅ Use custom schema instead of auto-generated
    prompt = CYPHER_GENERATION_PROMPT.format(schema=graph.get_schema, question=question)
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
    print("\n========= Generated Answer=========\n")
    print(cypher_query)

    result = chain_result.get("result", None)
    print("\n========= Final Result =========\n")
    print(json.dumps(chain_result, indent=2))

    return chain_result
