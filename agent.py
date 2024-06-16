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

#####################################################
### IMPORTS
def pdf_to_agent(pdf_doc) -> ReActAgent:
    # Save into Storage
    ss.storage_ctx.docstore.add_documents(pdf_doc)

    index = VectorStoreIndex.from_documents(
        pdf_doc,
        storage_context=ss.storage_ctx,
        # service_context=ss.service_ctx,
        use_async=True
    )

    # Query Engine
    base_query_engine = index.as_query_engine(
        similarity_top_k=3,
        # , "filters": filters
        use_async=True
    )
    response_synth = get_response_synthesizer(response_mode='compact')

    # Query Engine Tools
    sqe_tools = [
        QueryEngineTool(
            query_engine=base_query_engine,
            metadata=ToolMetadata(
                name="Document_Query_Engine",
                description="""A query engine that can answer a question about the user-submitted document. 
Single questions about the document should be asked here.""".strip()
            ),
        )
    ]

    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=sqe_tools,
        # service_context=ss.service_ctx,
        response_synthesizer=response_synth,
        use_async=True,
    )

    # Agent Tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="Document_Question_Engine",
                description="""A query engine that can answer questions about a user-submitted document.
Any questions about the document should be asked here.""".strip()
            ),
        )
    ]

    # Agent
    agent = ReActAgent.from_tools(
        tools=query_engine_tools, # type: ignore
        llm=ss.llm,
