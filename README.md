---
title: Autodoc Lifter
emoji: 🦊📝
colorFrom: white
colorTo: orange
sdk: streamlit
app_file: app.py
pinned: true
---

# Autodoc-Lifter
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