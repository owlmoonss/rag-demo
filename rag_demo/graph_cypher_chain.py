import json
import logging
import streamlit as st
from retry import retry
from langchain.chains import GraphCypherQAChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

CYPHER_GENERATION_TEMPLATE = """
You are a Cypher expert who translates natural language questions into Cypher queries for a Neo4j graph database. 
The database contains entities such as:
- Paper (p)
- Location (l)
- OceanCirculation (oc)
- WeatherEvent (we)
- Teleconnection (t)
- Model or Project (m)

Relationships include:
- :Mention (from Paper to another node), with property: Mention_Sentence
- :TargetsLocation (from a concept like OceanCirculation or WeatherEvent to a Location)

Properties include:
- Name (for all nodes)
- Mention_Sentence (in the :Mention relationship)
- wikidata_description (for Location)

- Ocean circulation processes often target specific oceanic locations, such as the Southern Ocean.
- Mentions of concepts in papers are linked via the :Mention relationship, which includes a Mention_Sentence field.
- To filter by concepts like "upwelling", check if the Mention_Sentence contains that word.
- A common type of question is: "What [scientific concept] are discussed in relation to [location] and involving [mechanism/phenomenon]?"
- The Cypher query often starts by matching a domain concept (e.g., OceanCirculation) and the location it's associated with.
- Then it retrieves papers mentioning that concept, filtering by keywords in the mention sentence.
When a question involves both a *scientific process* (e.g., upwelling) and a *geographical region* (e.g., Southern Ocean), first match entities representing the process (like OceanCirculation) that are linked to the region using :TargetsLocation. Then find papers mentioning those processes where the mention text contains the target concept (e.g., upwelling).
Some questions require understanding the relationship between a scientific process and a region, like the Southern Ocean. OceanCirculation entities are typically linked to Location nodes via the :TargetsLocation relation. If a specific process like "upwelling" is involved, the :Mention relationship's Mention_Sentence field can be used to filter relevant context in papers. Use this reasoning to write Cypher queries.
Entities like OceanCirculation, Teleconnection, and Paper are connected via various relationships. Each Paper can mention a concept using a :Mention relationship, which includes a field called Mention_Sentence.

Use Mention_Sentence to filter for specific keywords or phenomena that appear in the paper's context. For example, when a user asks about a certain process like “upwelling,” you can search for papers that mention that process by filtering the Mention_Sentence field.

Also, many scientific processes (e.g., OceanCirculation) are geographically grounded. You can find such processes using the :TargetsLocation relationship with a Location node. Locations often include names like “Southern Ocean”, “North Atlantic”, or “Southeast United States”.

If the question refers to regions or physical processes, combine both semantic filtering via Mention_Sentence and geographic filtering using TargetsLocation.

Important: Never use [:Mention] in query and Name of Location always uppercase and replace space with _ (example "North Atlantic" becomes "NORTH_ATLANTIC").
The following is the schema of the Neo4j database. The schema is a simplified representation of the graph database, showing the types of nodes and relationships present in the database. The schema includes nodes for Paper, Location, OceanCirculation, WeatherEvent, Teleconnection, and Model or Project, along with their respective properties and relationships.


Schema database is:
{schema}

Here are some examples:

### Example 1
Natural Language Question:
Which papers mention anomalous temperature regimes such as cold air outbreaks (CAOs) or warm waves (WWs) in relation to North America, specifically in the sentences where these terms appear?

Cypher:
MATCH (we)-[:TargetsLocation]-(l{{Name:"NORTH_AMERICA"}}) 
MATCH (p:Paper)-[m:Mention]-(we) 
WHERE m.Mention_Sentence CONTAINS 'WW' OR m.Mention_Sentence CONTAINS 'CAOs' 
RETURN p,l,we;

---

### Example 2
Natural Language Question:
Which papers discuss ocean circulation processes—such as thermohaline circulation—in oceanic regions that include either “North” or “South” in their names?

Cypher:
MATCH (n:Location) 
WHERE n.Name CONTAINS 'OCEAN' AND (n.Name CONTAINS 'NORTH' OR n.Name CONTAINS 'SOUTH') 
MATCH (oc:OceanCirculation)-[:TargetsLocation]-(l) 
MATCH (p:Paper)-[m:Mention]-(oc) 
WHERE m.Mention_Sentence CONTAINS 'thermohaline circulation' 
RETURN n,oc,p;

---

### Example 3
Natural Language Question:
Which papers mention CMIP5 models and the North Atlantic Oscillation (NAO) in the context of the Southeast United States?

Cypher:
MATCH (p:Paper)-[r:Mention]->(m:Model|Project) 
WHERE m.Name CONTAINS 'CMIP_5' 
MATCH (p)-[t:Mention]-(n:Teleconnection{{Name:"NORTH_ATLANTIC_OSCILLATION"}}) 
WHERE t.Mention_Sentence CONTAINS 'Southeast' 
RETURN p,m,n;

---

### Example 4
Natural Language Question:
Which papers mention the Pacific-North American (PNA) pattern in connection with locations in the United States?

Cypher:
MATCH (p:Paper)-[z:Mention]->(t:Teleconnection{{Name:"PACIFIC_NORTH_AMERICAN_PNA_PATTERN"}}) 
MATCH (t)-[:TargetsLocation]-(l:Location) 
MATCH (p)-[z:Mention]-(l) 
WHERE l.wikidata_description CONTAINS "United States" 
RETURN p,t,l;

---

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
    # cypher_llm=ChatOllama(model="qwen2", temperature=0),
    # qa_llm=ChatOllama(model="qwen2", temperature=0),
    cypher_llm=ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"], 
        temperature=0, 
        model_name="gpt-4o-mini"
    ),
    qa_llm=ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"], 
        temperature=0, 
        model_name="gpt-4o-mini"),
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
        print('No answer was generated.')
        return "No answer was generated."

    # ✅ Debug: show Cypher used
    cypher_query = chain_result.get("cypher", "No Cypher returned")
    print("\n========= Generated Answer=========\n")
    print(cypher_query)

    result = chain_result.get("result", None)
    print("\n========= Final Result =========\n")
    print(json.dumps(chain_result, indent=2))

    return chain_result
