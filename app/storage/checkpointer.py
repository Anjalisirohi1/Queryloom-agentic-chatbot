import os
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

conn = psycopg.connect(
    os.getenv("DATABASE_URL"),
    autocommit=True,
    prepare_threshold=0,
    row_factory=dict_row
)

checkpointer = PostgresSaver(conn)
checkpointer.setup()