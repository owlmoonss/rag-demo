from retry import retry
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from graph_cypher_tool import graph_cypher_tool
import streamlit as st

# llm = Ollama(model="llama3")
llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.2,
    model_name="gpt-4o-mini"
)
conversation_history = []

def process_with_llm(question: str) -> str:
    """Decide if a DB query is needed, and generate a response accordingly."""
    
    # === 1. Prepare conversation history (if any) ===
    conversation_text = "\n".join([
        f"User: {msg['input']}\nBot: {msg['output']}"
        for msg in conversation_history
    ])
    
    tool_output = graph_cypher_tool.invoke(question)
    tool_output_str = str(tool_output)
        
    # Send full prompt to process queried data
    final_prompt = f"""
Based on the conversation and the user question, provide a relevant and helpful response.

Conversation:
{conversation_text}

Current question: {question}

Here is the output from the database:
{tool_output_str}

Please process the output and answer the user question clearly.
    """.strip()
        
    final_response = llm.predict(final_prompt).strip()
    
    # === 5. Update conversation history ===
    conversation_history.append({
        "input": question,
        "output": final_response
    })
    
    return final_response

@retry(tries=2, delay=10)
def get_results(question: str) -> dict:
    llm_processed_output = process_with_llm(question=question)

    # 3. Return the result including both the tool output and the LLM response
    return {
        "input": question,  # User input
        "output": llm_processed_output,  # Output after processing with LLM
        # "intermediate_steps": tool_result["intermediate_steps"],  # Intermediate steps (if any)
        # "tools_used": tool_result["tools_used"],  # List of tools used
    }
