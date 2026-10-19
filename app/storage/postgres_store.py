import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from langgraph.store.postgres import PostgresStore

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise RuntimeError("DATABASE_URL is missing from .env")

store_conn = psycopg.connect(
    database_url,
    autocommit=True,
    prepare_threshold=0,
    row_factory=dict_row,
)

store = PostgresStore(store_conn)
store.setup()