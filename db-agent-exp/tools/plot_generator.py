import base64
import uuid
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.io as pio
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

from .python_tool import get_python_lc_tool

PLOT_INIT_CODE = """
import pandas as pd
import plotly.graph_objects as go
"""


class PlotGenerator:
    def __init__(self, source_python_tool: Callable, plot_folder: Path):
        self.source_python_tool = source_python_tool
        self.plot_folder = Path(plot_folder)
        self.source_local = self.source_python_tool.func.__self__.locals

    def generate_plot(self, plot_instruction, variable_name, query=None):
        if variable_name not in self.source_local:
            return {
                "observation": f"{variable_name} not found in the python environment",
                "metadata": {
                    "error": True,
                },
            }

        df = self.source_local[variable_name]

        if not isinstance(df, pd.DataFrame):
            return {
                "observation": f"{variable_name} is not a pandas dataframe",
                "metadata": {
                    "error": True,
                },
            }

        plot_system_prompt = """
Write python code to generate the plot using plotly

pre-executed code:
>>>
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv(csv_path)
<<< 

df.sample:
----
{df_sample}

df.dtype
----
{df_dtype}

len(df)
----
{df_len}


execute python code to create plotly plot using the instruction provided

Take extra care in making the plot aesthetically pleasing
Always preser light theme for the charts
Only create the plot object do not show the plot (i.e do not execute .show())
Do not attempt to create the variable `df` again, start execution after the pre-executed code.

return JSON output to 2 keys `code` & `plot_variable`

plot_variable: name of python variable containing the plot object
"""

        df_len = len(df)
        sample_size = min(df_len, 10)
        plot_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", plot_system_prompt),
                ("human", "{input}"),
            ]
        ).partial(
            df_sample=df.sample(sample_size).to_string(),
            df_len=df_len,
            df_dtype=df.dtypes.to_csv(header=False),
        )

        chain = (
            plot_prompt_template
            | ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            | JsonOutputParser()
        )
        plot_gen_result = chain.invoke({"input": plot_instruction})
        plot_py_tool = get_python_lc_tool(
            self.plot_folder, init_code=PLOT_INIT_CODE, reject_plot=False
        )
        plot_py_tool.func.__self__.locals["df"] = self.source_local[variable_name]
       
        plot_code_result = plot_py_tool.invoke({"code": plot_gen_result["code"]})
        print("Code:")
        print(plot_gen_result["code"])
        if plot_code_result.get("metadata", {}).get("error"):
            # TODO: Refactor dubplicate code
            plot_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", plot_system_prompt),
                    ("human", "{input}"),
                    (
                        "ai",
                        f"Error: {plot_code_result['observation']}\nFix the error and try again",
                    ),
                ]
            ).partial(
                df_sample=df.sample(sample_size).to_string(),
                df_len=df_len,
                df_dtype=df.dtypes.to_csv(header=False),
            )
            chain = (
                plot_prompt_template
                | ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    model_kwargs={"response_format": {"type": "json_object"}},
                )
                | JsonOutputParser()
            )
            plot_gen_result = chain.invoke({"input": plot_instruction})
            plot_code_result = plot_py_tool.invoke({"code": plot_gen_result["code"]})
            if plot_code_result.get("metadata", {}).get("error"):
                return {
                    "observation": f"Failed to generate plot with error: {plot_code_result['observation']}",
                    "metadata": {"error": True},
                }
        fig = plot_py_tool.func.__self__.locals[plot_gen_result["plot_variable"]]
        image_path = self.plot_folder / f"{str(uuid.uuid4())}.jpg"
        pio.write_image(fig, image_path, format="jpg")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        if not query:
            return {
                "observation": None,
                "metadata": {
                    "image_path": image_path,
                    "plot_gen_code": plot_gen_result["code"],
                },
            }

        plot_observation_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Assume the user is blind & generate observation for the question based on graph image",
                ),
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                        {"type": "text", "text": "{query}"},
                    ]
                ),
            ]
        )
        chain = (
            plot_observation_prompt_template
            | ChatOpenAI(model="gpt-4o", temperature=0)
            | StrOutputParser()
        )
        result = chain.invoke({"query": query})
        return {
            "observation": result,
            "metadata": {
                "image_path": image_path,
                "plot_gen_code": plot_gen_result["code"],
            },
        }


class PlotGeneratorInputs(BaseModel):
    plot_instruction: str = Field(
        description="instruction of the kind of plot we are looking to generate"
    )
    variable_name: str = Field(
        description="which variable to use as source table to the python environment"
    )


class PlotGeneratorInputsWithQuery(BaseModel):
    query: str = Field(
        description="what question we are trying to answer from the plot"
    )


def get_plot_gen_lc_tool(
    source_python_tool: Callable, plot_folder: Path, with_query=True
):
    plot_generator = PlotGenerator(
        source_python_tool=source_python_tool, plot_folder=plot_folder
    )
    description = "Tool to generate plot of a dataframe using the instructrion provided"
    return StructuredTool.from_function(
        plot_generator.generate_plot,
        name="plot_generator",
        description=description,
        args_schema=PlotGeneratorInputsWithQuery if with_query else PlotGeneratorInputs,
    )
