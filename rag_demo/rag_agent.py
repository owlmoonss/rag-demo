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
    
    # === 1. Chuẩn bị phần hội thoại lịch sử (nếu có) ===
    conversation_text = "\n".join([
        f"User: {msg['input']}\nBot: {msg['output']}"
        for msg in conversation_history
    ])
    
    tool_output = graph_cypher_tool.invoke(question)
    tool_output_str = str(tool_output)
        
    # Gửi prompt đầy đủ để xử lý dữ liệu đã truy vấn
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
    
    # === 5. Cập nhật lịch sử hội thoại ===
    conversation_history.append({
        "input": question,
        "output": final_response
    })
    
    return final_response

@retry(tries=2, delay=10)
def get_results(question: str) -> dict:
    llm_processed_output = process_with_llm(question=question)

    # 3. Trả về kết quả bao gồm cả đầu ra từ tool và kết quả xử lý của LLM
    return {
        "input": question,  # Đầu vào từ người dùng
        "output": llm_processed_output,  # Đầu ra sau khi xử lý bằng LLM
        # "intermediate_steps": tool_result["intermediate_steps"],  # Các bước trung gian (nếu có)
        # "tools_used": tool_result["tools_used"],  # Danh sách các tool đã sử dụng
    }
