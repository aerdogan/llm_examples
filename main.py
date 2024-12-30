from functools import lru_cache
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

graph = StateGraph(MessagesState)

@lru_cache(maxsize=None)
def factorial(n: int) -> int:
    return n * factorial(n - 1) if n else 1

def calculate_factorial(n: int) -> int:
    """n'nin faktöriyelini hesaplayın

    Parametreler:
        n: Faktöriyelini hesaplayacak değer

    Sonuç:
        int: Faktöriyel hesaplamanın sonucu
    """
    return factorial(n)

def multiply(a: int, b: int) -> int:
    """Giriş a ile b çarpımını verir

    Parametreler:
        a: ilk sayı
        b: ikinci sayı

    Sonuç:
        int: Çarpım sonucu
    """
    return a * b

def addition(a: int, b: int) -> int:
    """Giriş a ile b toplamını verir

    Parametreler:
        a: ilk sayı
        b: ikinci sayı

    Sonuç:
        int: Toplama sonucu
    """
    return a + b

tools = [addition, multiply, calculate_factorial]
model = ChatOllama(model="qwen2.5-coder")
model_with_tools = model.bind_tools(tools)

def math_llm(state):
    msg_content = "Sen profesyonel bir matematik asistanısın. Sorulan işlemleri yapmak için her aşamada işlem araçlarını kullanmalısın."
    message = [SystemMessage(content=msg_content)] + state['messages']
    return {"messages": model_with_tools.invoke(message)}

graph.add_node("math_llm", math_llm)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "math_llm")
graph.add_edge("tools", "math_llm")
graph.add_edge("tools", END)
graph.add_conditional_edges("math_llm", tools_condition)

agent = graph.compile()

msg_content = "5 faktöriyelini hesapla, 3 ile çarp, sonra 8 ekle"
state = {"messages": [HumanMessage(msg_content)]}

resp = agent.invoke(state)
for message in resp['messages']:
    message.pretty_print()
