from retry import retry
from langchain_community.llms import Ollama
from graph_cypher_tool import graph_cypher_tool

llm = Ollama(model="llama3")

conversation_history = []

def process_with_llm(question, json_output: any) -> str:
    """Send the tool output (JSON) to the LLM for further processing."""
    # Chuyển kết quả JSON thành chuỗi, bạn có thể format hoặc làm sạch theo yêu cầu
    output_str = str(json_output)
    conversation_text = "\n".join([f"User: {msg['input']}\nBot: {msg['output']}" for msg in conversation_history])
    prompt = f"Based on the conversation, provide a relevant and helpful response.\n{conversation_text}\n\n"
    prompt = prompt + f"""
This is current question by user: {question}
Please process the output and provide a summary or answer based on the data.
Make sure to answer the question in a clear and concise manner.
Here is the output from the database:
{json_output}
    """

    # Gửi kết quả tới LLM để xử lý
    response = llm.predict(prompt)
    
    conversation_history.append({"input": question, "output": response})
    
    return response


@retry(tries=2, delay=10)
def get_results(question: str, user_args: dict = None, tool_name: str = None) -> dict:
    """
    Run a specific tool directly, then send the result to LLM for processing.

    Args:
        question (str): User’s original question (for record only)
        user_args (dict): Dict input for the tool
        tool_name (str): Name of the tool to use (must be in custom_tools)

    Returns:
        dict: agent-style output, with LLM processed result
    """
    tool = graph_cypher_tool
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found.")

    # 1. Chạy tool để lấy kết quả
    tool_result = graph_cypher_tool(question)

    # 2. Gửi kết quả tool (JSON) đến LLM để xử lý thêm
    llm_processed_output = process_with_llm(question=question, json_output=tool_result)

    # 3. Trả về kết quả bao gồm cả đầu ra từ tool và kết quả xử lý của LLM
    return {
        "input": question,  # Đầu vào từ người dùng
        "output": llm_processed_output,  # Đầu ra sau khi xử lý bằng LLM
        # "intermediate_steps": tool_result["intermediate_steps"],  # Các bước trung gian (nếu có)
        # "tools_used": tool_result["tools_used"],  # Danh sách các tool đã sử dụng
    }
