from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

# 사용할 db 경로로 수정
db = SQLDatabase.from_uri('sqlite:///Chinook.db')
print(db.get_usable_table_names())

llm = ChatOllama(model=gemma_model, temperature=0)

# 질문을 SQL 쿼리로 변환
write_query = create_sql_query_chain(llm, db)

# SQL 쿼리 실행
execute_query = QuerySQLDatabaseTool(db=db)

# combined chain = write_query | execute_query
combined_chain = write_query | execute_query

# 체인 실행
result = combined_chain.invoke({'question': '직원(employee)은 모두 몇 명인가요?'})

print(result)