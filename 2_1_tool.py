from langchain import LlamaCpp
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool


def multiplier(a: float, b: float) -> float:
    """Multiply the provided floats."""
    print("Tool used to multiply {0} times {1}".format(a, b))
    return a * b

tool = StructuredTool.from_function(multiplier)

llm = LlamaCpp(
    model_path='./model/wizardLM-7B.ggmlv3.q5_1.bin',
    n_gpu_layers=40,
    n_ctx=2048,
    temperature=0
)

agent_executor = initialize_agent(
    [tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent_executor.run("What is 5433 times 765")
