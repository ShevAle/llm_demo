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

instructions = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only one $JSON_BLOB and only one action per $JSON_BLOB, as shown:
```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:
Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""

suffix = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. 
Use tools if necessary. 
Format is Question: than Thought: than Action: ```$JSON_BLOB``` then Observation:."""

agent_executor = initialize_agent(
    [tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={
        "format_instructions": instructions,
        "suffix": suffix
    },
    verbose=True
)

agent_executor.run("What is 5433 times 765")
