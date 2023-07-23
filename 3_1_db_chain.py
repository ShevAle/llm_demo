from langchain import LlamaCpp, SQLDatabase, SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///./db/sqlite/Chinook.db",
                          sample_rows_in_table_info=0)

llm = LlamaCpp(
    model_path='./model/wizardLM-7B.ggmlv3.q5_1.bin',
    n_gpu_layers=40,
    n_ctx=2048,
    temperature=0
)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

db_chain("How many tracks that have word 'man' in their name and longer than 300000 milliseconds are there?")
