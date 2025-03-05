import typing as t
from error import PipelineError
from pipeline import Pipeline, RowLevelPipelineError, pipeline_error_handler
from crewai_tools import CodeDocsSearchTool, SerperDevTool, ScrapeWebsiteTool
import requests


class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        input_fields: t.List[str],
        output_fields: t.List[str],
        parameters: t.Dict[str, t.Any],
        failure_modes: t.List[str],
        run_method: t.Callable[..., t.Tuple],
    ):
        self.name = name
        self.description = description
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.parameters = parameters
        self.failure_modes = failure_modes
        self.run_method = run_method

    def run(self, *inputs) -> t.Tuple:
        print("Running agent", self.name, "with inputs", inputs)
        return self.run_method(*inputs)


def SearchSerper(query: str):
    tool = SerperDevTool()
    return tool.run(search_query=query)


SearchSerperAgent = Agent(
    name="SearchSerper",
    description="Search Google for the given query",
    input_fields=["natural language query"],
    output_fields=["json of relevant google search results information and links"],
    parameters={},
    failure_modes=[],
    run_method=SearchSerper,
)


def SearchCodeDocs(query: str, docs_url: str):
    tool = CodeDocsSearchTool(docs_url=docs_url)
    return tool.run(query)


SearchCodeDocsAgent = Agent(
    name="SearchCodeDocs",
    description="Search the given/linked code docs for the given query",
    input_fields=["query", "docs_url"],
    output_fields=["summary of relevant information"],
    parameters={},
    failure_modes=[],
    run_method=SearchCodeDocs,
)


def ScrapeWebsite(website_url: str):
    tool = ScrapeWebsiteTool()
    return tool.run(website_url=website_url)


ScrapeWebsiteAgent = Agent(
    name="ScrapeWebsite",
    description="Scrape the website for the given url",
    input_fields=["website_url"],
    output_fields=["text of website content"],
    parameters={},
    failure_modes=[],
    run_method=ScrapeWebsite,
)
