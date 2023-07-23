from langchain import LlamaCpp, SQLDatabase, SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///./db/sqlite/Chinook.db",
                          sample_rows_in_table_info=0)

llm = LlamaCpp(
    model_path='./model/llama-2-13b.ggmlv3.q4_1.bin',
    n_gpu_layers=30,
    n_ctx=4096,
    temperature=0
)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

db_chain("How many tracks that have word 'man' in their name and longer than 300000 milliseconds are there?")
