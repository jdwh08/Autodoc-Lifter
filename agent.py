#####################################################
### DOCUMENT PROCESSOR [AGENT]
#####################################################
### Jonathan Wang

# ABOUT: 
# This creates an app to chat with PDFs.

# This is the AGENT
# which handles complex questions about the PDF.
#####################################################
### TODO Board:
# https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/agent_runner_rag_controllable/#setup-human-in-the-loop-chat
# Investigate ObjectIndex and retrievers? https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents/
# https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores/

#####################################################
### IMPORTS
from typing import List

from streamlit import session_state as ss

from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

# Own Modules
from full_doc import FullDocument

#####################################################
### CODE

ALLOWED_DOCUMENT_TOOLS = ['engine', 'subquestion_engine']
ALLOWED_TOOLS = ALLOWED_DOCUMENT_TOOLS

def _build_tool_from_fulldoc(fulldoc: FullDocument, tool_name: str) -> QueryEngineTool:
        """Given a Full Document, build a QueryEngineTool from the specified engine.

        Args:
            fulldoc (FullDocument): The FullDocument (doc + query engines)
            tool_name (str): The engine to use.

        Returns:
            QueryEngineTool: A query engine wrapper around the tool.
        """
        if (tool_name.lower() not in ALLOWED_DOCUMENT_TOOLS):
            raise ValueError("`tool_name` must be one of {ALLOWED_DOCUMENT_TOOLS}")
        if (getattr(fulldoc, tool_name, None) is None):
            raise ValueError(f"`{tool_name}` must be created from the document first.")
        
        # Build Tool
        tool_description = ''
        if tool_name == 'engine':
            tool_description += 'A tool that answers simple questions about the following document:\n' + fulldoc.summary_oneline
        elif tool_name == 'subquestion_engine':
            tool_description += 'A tool that answers complex questions about the following document:\n' + fulldoc.summary_oneline
        
        tool = QueryEngineTool(
            query_engine=getattr(fulldoc, tool_name),
            metadata=ToolMetadata(
                name=tool_name,
                description=tool_description
            ),
        )
        return tool

def doclist_to_agent(doclist: List[FullDocument], fulldoc_tools_to_use: List[str]=['engine']) -> SubQuestionQueryEngine: # ReActAgent:
    # Agent Tools
    agent_tools = []
    
    # Remove any tools that are not in the allowed list using
    tools_to_use = list(set(fulldoc_tools_to_use).intersection(set(ALLOWED_DOCUMENT_TOOLS)))
    if (len(tools_to_use) < len(fulldoc_tools_to_use)):
        removed_tools = set(fulldoc_tools_to_use) - set(ALLOWED_DOCUMENT_TOOLS)
        Warning(f"Tools {removed_tools} are not in the allowed list of tools. Skipping...")
        del removed_tools
    
    for tool in tools_to_use:
        for doc in doclist:
            agent_tools.append(_build_tool_from_fulldoc(doc, tool))

    # Agent
    # agent = ReActAgent.from_tools(
    agent = SubQuestionQueryEngine.from_defaults(
        # tools=agent_tools,
        query_engine_tools=agent_tools,
        llm=Settings.llm or ss.llm,
        verbose=True,
        # max_iterations=5
    )

    return agent
