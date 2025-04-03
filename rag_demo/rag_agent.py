from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain import hub  # requires langchainhub package
from pydantic_ai.models.openai import OpenAIModel as OpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from vector_graph_tool import vector_graph_tool
from graph_cypher_tool import graph_cypher_tool
from graph_tool import graph_tool
from vector_tool import vector_tool
from retry import retry
from langchain_ollama import ChatOllama



# Setup tools the agent will use
llm = ChatOllama(
        model="qwen2",  # Replace with the appropriate Ollama model
        temperature=0,
    )
tools = load_tools([], llm=llm)
tools = tools + [graph_cypher_tool, graph_tool]


# REACT AGENT EXECUTOR

prompt = hub.pull("hwchase17/react")

# More on agent types: https://python.langchain.com/docs/modules/agents/agent_types/
agent = create_react_agent(llm, tools, prompt)

# NOTE: early_stopping_method generate option ONLY available for multi-action agents
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


@retry(tries=2, delay=20)
def get_results(question, callbacks) -> dict:
    """
    Starts a LangChain agent to generate an answer using one of several Neo4j RAG tools.

    Args:
        question (str): User query
        callbacks (list): List of optional LangChain callback handlers

    Returns:
        dict: Final answer as a dict with the keys: input, output, intermediate_steps
    """
    response = agent_executor.invoke({"input": question}, callbacks=callbacks)

    # Extract more info from the response
    full_input = response.get("input")
    final_output = response.get("output")
    intermediate_steps = response.get("intermediate_steps")

    # Optional: extract tool names used
    tools_used = [step[0] for step in intermediate_steps] if intermediate_steps else []

    return {
        "input": full_input,
        "output": final_output,
        "intermediate_steps": intermediate_steps,
        "tools_used": tools_used,
    }


