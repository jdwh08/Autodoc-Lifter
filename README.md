---
title: Autodoc Lifter
emoji: 🦊📝
colorFrom: yellow
colorTo: red
python_version: 3.11.9
sdk: streamlit
sdk_version: 1.37.1
suggested_hardware: t4-small
suggested_storage: small
app_file: app.py
header: mini
short_description: Good Local RAG for Bad PDFs
models: [timm/resnet18.a1_in1k, microsoft/table-transformer-detection, mixedbread-ai/mxbai-embed-large-v1, mixedbread-ai/mxbai-rerank-large-v1, meta-llama/Meta-Llama-3.1-8B-Instruct, Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5]
tags: [rag, llm, pdf, document]
license: agpl-3.0
pinned: true
preload_from_hub:
    - timm/resnet18.a1_in1k
    - microsoft/table-transformer-detection
    - mixedbread-ai/mxbai-embed-large-v1
    - mixedbread-ai/mxbai-rerank-large-v1
    - Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5
---

## Autodoc Lifter

Document RAG system with LLMs.
Some key goals for the project, once finished:

0. All open, all local.
    I don't want to be calling APIs. You can the entire app locally, and inspect the code and models.
    This is particularly suitable for handling restricted information.
    Yes I know this is a web demo on Spaces, so don't actually do that here. 
    Use the GitHub link: (here, once it's no longer ClosedAI)

1. Support for atrocious and varied PDFs.
    Have images? Have tables? Have a set of PDFs with the worst quality and page layout known to man? 
    Give it a try in here. I've been slowly building out custom processing for difficult documents by connecting Unstructured.IO to LlamaIndex in a slightly useful way. 
    (A future dream: get rid of Unstructured and build our own pipeline one day.)

2. Multiple PDFs, handled with agents.
    Instead of dumping all the documents into one central vector store and praying it works out,
    I'm try to be more thoughtful as to how to incorporate multiple documents.

3. Answers that are sourced and verifiable.
    I'm sorry, but as an Definitely Human Person, I don't like hallucinated answers-ex-machina.
    Responses should give actual citations \[0\] when pulling text directly from source documents,
    and there should be a way to view the citations, referenced text, and the document itself.

    --- CITATIONS ---
    \[0\] Relies primarily on fuzzy string matching, because it's computationally cheaper and also
    ensures that cited text actually occurs in the source documents.