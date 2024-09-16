import threading
import re
import ast
import astor
from loguru import logger
from pathlib import Path
import uuid
import pandas as pd
from typing import Dict, List
from io import StringIO
from contextlib import redirect_stdout
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
from rapidfuzz import fuzz


def convert_triple_quoted_to_single_line(code):
    """Converts triple quoted strings to single line strings."""
    tree = ast.parse(code)

    class TripleQuoteTransformer(ast.NodeTransformer):
        def visit_Str(self, node):
            if isinstance(node, ast.Constant) and isinstance(node.s, str):
                new_lines_str = node.s.replace("\n", "\\n")
                node.s = new_lines_str
            return node

    transformer = TripleQuoteTransformer()
    transformed_tree = transformer.visit(tree)

    return astor.to_source(transformed_tree)


# code syntax check
def is_correct_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def sanitize_python_code(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """
    # Removes `, whitespace & python from start
    # query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    # query = re.sub(r"(\s|`)*$", "", query)
    return query
    # return query.replace("\\n", "\n")


INSECURE_LIBRARIES = {
    "subprocess",
    "os",
    "eval",
    "pickle",
    "shlex",
    "sys",
    "tempfile",
    "popen",
    "pipe",
    "fcntl",
    "termios",
    "ctypes",
    "socket",
    "http.server",
    "http.client",
    "multiprocessing",
    "imp",
    "imp.load_source",
    "requests",
    "xml.etree.ElementTree",
    "urllib.request",
    "urllib.urlopen",
    "urllib.urlretrieve",
}

INSECURE_FUNCTIONS = {
    "exec",
    "eval",
    "pickle.loads",
    "input",
    "__import__",
    "os.system",
    "sh",
    "system",
    "getpass",
    "fileinput",
    "imaplib",
    "poplib",
    "smtplib",
    "telnetlib",
    "ftplib",
    "http.client",
    "http.server",
    "sqlite3",
    "mysql.connector",
    "paramiko",
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.check_output",
    "subprocess.getstatusoutput",
    "os.popen",
    "os.spawn",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.environ",
}


def extract_modules_functions(code: str):
    """Extracts all modules and functions from the code
    Return function as full name if it is a method of a module

    Args:
        code (str): The code to extract modules and functions from

    Returns:
        tuple: A tuple containing the modules and functions
    """
    # Extracts all modules and functions from the code
    modules = set()
    functions = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                # if node.func.value is instance of ast.Call it does not have attribute id, (so node.func.value.id will throw attribute error)
                if isinstance(node.func.value, ast.Name):
                    functions.add(f"{node.func.value.id}.{node.func.attr}")
            elif isinstance(node.func, ast.Name):
                functions.add(node.func.id)
    return modules, functions


def is_code_insecure(code: str) -> bool:
    """
    Check if the code is insecure
    - Check for insecure libraries and functions
    - log the insecure libraries or function detected
    """
    modules, functions = extract_modules_functions(code)
    insecure_modules = modules.intersection(INSECURE_LIBRARIES)
    insecure_functions = functions.intersection(INSECURE_FUNCTIONS)
    if insecure_modules or insecure_functions:
        logger.warning(
            f"Insecure code detected: {insecure_modules or insecure_functions}"
        )
        return True
    return False


INIT_CODE = """
import nltk
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
import statsmodels
import re
import collections

"""


def get_csv_df_repr(df: pd.DataFrame) -> str:
    """
    Get a CSV representation of a pandas DataFrame.
    """
    return df.to_csv(index=False)


def get_init_variables(folder_path: Path):
    import nltk
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    import sklearn
    import statsmodels
    import re
    import collections

    pd.set_option("display.max_rows", 2000)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", None)

    # Return csv text while printing
    # pd.DataFrame.__repr__ = get_csv_df_repr
    # pd.DataFrame.__str__ = get_csv_df_repr

    return {
        # Overwrite plt to save the file on plt.show
        "np": np,
        "pd": pd,
        "sklearn": sklearn,
        "statsmodel": statsmodels,
        "st": st,
        "nltk": nltk,
    }


code_exec_lock = threading.Lock()


class PythonTool:
    def __init__(
        self,
        folder: Path,
        locals: Dict = None,
        globals: Dict = None,
        sanitize_input=True,
        init_code=None,
        reject_plot=True,
        reject_dataframe_creation=True,
        attempt_dataframe_creation_removal=True,
    ):
        self.folder = folder
        self.locals = locals or {}
        self.globals = globals or {}
        self.sanitize_input = sanitize_input
        self.reject_plot = reject_plot
        self.reject_dataframe_creation = reject_dataframe_creation
        self.attempt_dataframe_creation_removal = attempt_dataframe_creation_removal
        self.locals.update(get_init_variables(folder))
        self.execution_history = INIT_CODE
        if init_code:
            self.execute_code(init_code)
            self.update_execution_history(init_code)

    def execute_code(self, code):
        # print("executing code")
        with code_exec_lock:
            # try:
            #     # print("inside try")
            #     io_buffer = StringIO()
            #     # print(code)
            #     with redirect_stdout(io_buffer):
            #         # print("Inside redirect io")
            #         exec(code, self.globals, self.locals)
            #         return io_buffer.getvalue()
            # finally:
            #     # print("closing io buffer")
            #     io_buffer.close()
            io_buffer = StringIO()
            try:
                last_expr_str = None
                
                with redirect_stdout(io_buffer):
                    tree = ast.parse(code)
                    code_before_last_expr = ast.Module(
                        body=tree.body[:-1], type_ignores=[]
                    )
                    exec(ast.unparse(code_before_last_expr), self.locals, self.locals)
                    last_expr = ast.Module(body=[tree.body[-1]], type_ignores=[])
                    last_expr_str = ast.unparse(last_expr)
                    logger.info(f"Executing {last_expr_str}")
                    ret = eval(last_expr_str, self.globals, self.locals)
                    return io_buffer.getvalue(), ret
            except Exception as e:
                logger.warning(f"eval failed with with error {e} fo retrying with exec")
                logger.exception(e)
                if last_expr_str:
                    with redirect_stdout(io_buffer):
                        exec(last_expr_str, self.globals, self.locals)
                return io_buffer.getvalue(),None
            finally:
                io_buffer.close()

    def update_execution_history(self, code):
        self.execution_history += "\n" + code

    def create_dateframe_in_code(self, code):
        # return true if code creates a dataframe
        blacklisted_functions = {
            "pd.read_csv",
            "pd.read_excel",
            "pd.read_sql",
            "pd.DataFrame",
            "pd.read_sql_query",
        }
        _, detected_functions = extract_modules_functions(code)
        if detected_functions.intersection(blacklisted_functions):
            return True
        return False

    def plot_logic_in_code(self, code):
        # return true if code creates a plot
        blacklisted_modules = {"matplotlib", "seaborn", "plotly"}
        detected_modules, _ = extract_modules_functions(code)
        # filter submodules from detected_modules
        detected_modules = {module.split(".")[0] for module in detected_modules}
        if detected_modules.intersection(blacklisted_modules):
            return True
        return False

    def get_sql_dataframe_variables(self) -> List[str]:
        # return all variables that are pandas dataframe name is of pattern sql_result_df_{idx} or type of pandas dataframe
        return [
            var
            for var in self.locals
            if var.startswith("sql_result_df_")
            or isinstance(self.locals[var], pd.DataFrame)
        ]

    def filter_sql_dataframe_variables(self, code: str) -> str:
        # Todo: Use llm to generate the code
        # Extremely HACKY
        sql_dataframe_variables = self.get_sql_dataframe_variables()
        if not any(var in code for var in sql_dataframe_variables):
            return code
        code = convert_triple_quoted_to_single_line(code)
        # remove all the variables that are pandas dataframe name is of pattern sql_result_df_{idx}
        pattern = r"sql_result_df_\d+\s*=\s*pd\.read_sql_query\((?:.|\s)*?\)\s*\n"
        filtered_code = re.sub(pattern, "", code)
        return code

    def __call__(self, code: str):
        if self.sanitize_input:
            code = sanitize_python_code(code)

        execution_code = code
        metadata = {}

        try:
            if is_code_insecure(code):
                metadata["error"] = True
                return {
                    "observation": "Due to security reasons , we cannot execute this code (insecure imports or function calls)",
                    "metadata": metadata,
                }

            if self.reject_plot and self.plot_logic_in_code(code):
                metadata["error"] = True
                return {
                    "observation": "Code creates a plot, which is not allowed, please use the plot generator tool",
                    "metadata": metadata,
                }

            if self.reject_dataframe_creation and self.create_dateframe_in_code(code):
                logger.warning("Code creates a dataframe")
                sql_dataframe_variables = self.get_sql_dataframe_variables()
                filtered_code = self.filter_sql_dataframe_variables(code)
                if not is_correct_syntax(filtered_code):
                    logger.warning("Code syntax error after filtering")
                if (
                    not self.attempt_dataframe_creation_removal
                    or not is_correct_syntax(filtered_code)
                ):
                    return {
                        "observation": f"Code creates a dataframe, which is not allowed, please use one of the following pre-existing dataframe variables: {sql_dataframe_variables}, do not initialize/re-initialze it as it already exists",
                        "metadata": {"error": True},
                    }
                execution_code = filtered_code
                if filtered_code != code:
                    logger.warning("Code transformed due to dataframe creation")

            metadata["executed_code"] = execution_code
            print(execution_code)
            str_log, ret = self.execute_code(execution_code)
            self.update_execution_history(code)
            metadata["ret"] = ret
            metadata["str_log"] = str_log
            if isinstance(ret, pd.DataFrame):
                metadata["table"] = ret
            observation = str_log
            if ret is not None:
                observation += str(ret)
            return {"observation": observation, "metadata": metadata}
        except Exception as e:
            metadata["error"] = True
            # save stack trace in a variable as string
            logger.exception(e)
            return {
                "observation": f"{e.__class__.__name__}: {e}",
                "metadata": metadata,
            }


def find_matching_values(value, column_name, df):
    uniques = pd.Series(df[column_name].unique())
    uniques_scores = uniques.apply(
        lambda x: fuzz.partial_ratio(x.lower(), value.lower())
    )
    uniques_scores.index = uniques
    return uniques_scores.sort_values(ascending=False).head(5).index.to_list()


class ColumnMatchFinder:
    def __init__(self, source_python_tool):
        self.source_python_tool = source_python_tool
        self.source_local = self.source_python_tool.func.__self__.locals

    def match_column(self, value, column_name, df_name):
        if df_name not in self.source_local:
            return {
                "observation": f"{df_name} not found in the python environment",
                "metadata": {
                    "error": True,
                },
            }

        df = self.source_local[df_name]
        if not isinstance(df, pd.DataFrame):
            return {
                "observation": f"{df_name} is not a pandas dataframe",
                "metadata": {
                    "error": True,
                },
            }

        if column_name not in df.columns:
            return {
                "observation": f"{column_name} not found in the dataframe",
                "metadata": {
                    "error": True,
                },
            }

        matched_values = find_matching_values(value, column_name, df)
        return {
            "observation": str(matched_values),
            "metadata": {
                "error": False,
            },
        }


class PythonInputs(BaseModel):
    """Python inputs."""

    code: str = Field(description="code snippet to run(can be multiline)")


def get_python_lc_tool(folder: Path, init_code: str = None, **kwargs):
    py_tool = PythonTool(folder=folder, init_code=init_code, **kwargs)
    description = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "When using this tool, sometimes output is abbreviated - "
        "make sure it does not look abbreviated before using it in your answer."
    )
    return StructuredTool.from_function(
        py_tool.__call__,
        name="python_env",
        description=description,
        args_schema=PythonInputs,
    )


class MatchInput(BaseModel):
    value: str = Field(description="value to match")
    column_name: str = Field(description="column to match on")
    df_name: str = Field(description="dataframe which has the column")


def get_matching_values_tool(py_tool):
    description = (
        "Find matching values in a column of a dataframe. "
        "This tool uses fuzzy matching to find the closest matches to the input value."
    )
    match_finder = ColumnMatchFinder(py_tool)
    return StructuredTool.from_function(
        match_finder.match_column,
        name="find_matching_values",
        description=description,
        args_schema=MatchInput,
    )
