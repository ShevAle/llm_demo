from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import ctypes

os.environ["GGML_OPENCL_PLATFORM"] = "AMD"
os.environ["GGML_OPENCL_DEVICE"] = "1"

ctypes.cdll.LoadLibrary("d:/Projects/llm_demo/CLBlast-1.6.1-windows-x64/lib/clblast.dll")
ctypes.cdll.LoadLibrary("c:/vcpkg/packages/opencl_x64-windows/bin/OpenCL.dll")

template = """Question: {question}
Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="./model/llama-2-13b.ggmlv3.q4_1.bin",
    n_gpu_layers=50,
    callback_manager=callback_manager,
    temperature=0
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)
