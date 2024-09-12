from typing import Dict, List, Tuple, Callable
from abc import ABC, abstractmethod
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import OperationalError
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool


class BadConnectionURI(Exception):
    pass


class InvalidDBCredentials(BadConnectionURI):
    pass


class InvalidDBHost(BadConnectionURI):
    pass


class InvalidDBName(BadConnectionURI):
    pass


class DBEngineNotSupported(Exception):
    pass


# TODO: Filter table names
class DBEngine(ABC):

    @property
    @abstractmethod
    def db_type(self) -> str:
        """
        Get the type of the database
        """
        pass

    @abstractmethod
    def get_tables(self) -> List[str]:
        """
        Get all tables in the database
        """
        pass

    @abstractmethod
    def get_columns(self, table_name: str) -> List[Tuple[str, str]]:
        """
        Get all column along with types in the table
        """
        pass

    @abstractmethod
    def query_df(self, query: str) -> pd.DataFrame:
        """
        Execute a query and return the result as a DataFrame
        """
        pass

    @abstractmethod
    def table_sample(self, table_name: str, n: int = 5) -> pd.DataFrame:
        """
        Get a sample of the table
        """
        pass

    @abstractmethod
    def table_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in the table
        """
        pass


class SQAlchemyEngine(DBEngine):
    def __init__(self, connection_uri):
        self.connection_uri = connection_uri
        print("Creating engine")
        self.engine = create_engine(connection_uri)
        print("Connecting to database")
        # Handle connection failure
        self.connection = self.engine.connect()

    def get_tables(self) -> list[str]:
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_columns(self, table_name: str):
        inspector = inspect(self.engine)
        return [
            (d["name"], d["type"].__visit_name__)
            for d in inspector.get_columns(table_name)
        ]

    def sanitize_query(self, query: str) -> str:
        # escape % in the query
        return query.replace("%", "%%")

    def query_df(self, query: str) -> pd.DataFrame:
        query = self.sanitize_query(query)
        return pd.read_sql_query(query, self.connection)

    def table_sample(self, table_name: str, n: int = 5) -> pd.DataFrame:
        return pd.read_sql_query(
            f"SELECT * FROM {table_name} LIMIT {n}", self.connection
        )

    def table_row_count(self, table_name: str) -> int:
        return self.query_df(f"SELECT COUNT(*) FROM {table_name}").iloc[0, 0]

    def close(self):
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()

    def __del__(self):
        print("__del__ called for SQAlchemyEngine")
        self.close()


class MySQlEngine(SQAlchemyEngine):
    def __init__(self, connection_uri):
        try:
            super().__init__(connection_uri)
        except OperationalError as e:
            if "Access denied" in str(e):
                raise InvalidDBCredentials("Invalid credentials")
            if "Can't connect to MySQL server" in str(e):
                raise InvalidDBHost("Invalid host or port")
            if "Unknown database" in str(e):
                raise InvalidDBName("Invalid database name")
            raise BadConnectionURI(
                "Error initializing db connection, please check the connection details"
            )

    @property
    def db_type(self) -> str:
        return "mysql"


def get_csv_df_repr(df):
    return df.to_csv(index=False)


class SQLTool:
    def __init__(
        self, db_engine: DBEngine, python_tool: Callable = None, top_n: int = 20
    ):
        self.db_engine = db_engine
        self.python_tool = python_tool
        self.repl_vars = {}
        if self.python_tool:
            # TODO: Make sure python_tool is the right tool
            self.repl_vars = python_tool.func.__self__.locals
        self.top_n = top_n

    def query_df(self, sql_query: str) -> Dict:
        pd.DataFrame.__repr__ = lambda df: get_csv_df_repr(df)
        try:
            df = self.db_engine.query_df(sql_query)
        except Exception as e:
            return {
                "observation": f"{e.__class__.__name__}: {e}",
                "metadata": {"error": True},
            }
        df_name = self.get_new_df_name()
        self.repl_vars[df_name] = df
        if self.python_tool:
            # TODO: Simplify the python_tool method/member access logic
            self.python_tool.func.__self__.update_execution_history(
                f'{df_name} = pd.read_sql_query("""{sql_query}""")'
            )

        observation = (
            f"Below is {'the FIRST ' + str(self.top_n) + ' ROWS (Total rows=' +  str(len(df)) + ') of' if len(df) > self.top_n else ''} the generated sql result in csv format:\n"
            ">>>\n"
            f"{df.head(self.top_n).to_csv(index=False)}"
            "<<<\n"
            f"Result of SQL Query loaded in `{df_name}`:pd.Dataframe variable in python environment for further analysis"
        )
        return {
            "observation": observation,
            "metadata": {"data": df, "variable_name": df_name},
        }

    def get_new_df_name(self) -> str:
        """
        create a new name for the dataframe and make sure it does not clash with existing names in repl_vars dictionary
        """
        idx = 1
        while True:
            df_name = f"sql_result_df_{idx}"
            if df_name not in self.repl_vars:
                return df_name
            idx += 1


class SQLInputs(BaseModel):
    """SQL inputs."""

    sql_query: str = Field(description="A detailed and correct SQL query.")


SQLDBQUERY_DESCRIPTION = """
Input to this tool is a detailed and correct mysql query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again.
"""


def get_sql_lc_tool(
    db_engine: DBEngine, python_tool: Callable = None
) -> StructuredTool:
    sql_tool = SQLTool(db_engine, python_tool=python_tool)
    return StructuredTool.from_function(
        sql_tool.query_df,
        name="sql_workbench",
        description=SQLDBQUERY_DESCRIPTION,
        args_schema=SQLInputs,
    )
