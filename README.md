# LLM tutorials

This branch is targeted at playing around with Langchain and its capabilities and use cases

## Use Cases

### Chatbot with RAG over multiple documents

This tutorial comprises different incremental steps:

1. ask one question to a LLM
2. use a web loader to provide documents for RAG to ask one single question
3. (hardcoded) chat history
4. use a PDF loader for RAG
5. load multiple PDF documents from a folder
6. add a minimal UI class but keep the questions hardcoded (not interactive)
7. interactive chatbot, i.e., the user can enter questions and receive the answer
8. add loading multiple documents in different file formats. allows to define which folder to load, which system prompt to use as context and which set of predefined questions to use

Running the script, e.g.
  
    tutorial08-interactive-chat-RAG-multiple-filetypes.py docs/examples/DDD docs/questions/ddd.txt docs/prompts/ddd.txt 

## To do

- streaming response https://python.langchain.com/docs/use_cases/question_answering/streaming/
- tools 
  - https://python.langchain.com/docs/use_cases/tool_use/
  - https://python.langchain.com/docs/use_cases/chatbots/tool_usage/
  - https://blog.langchain.dev/tool-calling-with-langchain/
- agents
- trim chat history https://python.langchain.com/docs/use_cases/chatbots/memory_management/
- return sources https://python.langchain.com/docs/use_cases/question_answering/sources/
- per user retrieval https://python.langchain.com/docs/use_cases/question_answering/per_user/
- extract structured output https://python.langchain.com/docs/use_cases/extraction/
- Doctran interrogate document
  - https://python.langchain.com/docs/integrations/document_transformers/doctran_interrogate_document/
- better instructions for using Unstructured
    - what needs to be installed on the system, specially windows, besides the python library bindings in the virtual environment
- read a list of urls and load using the WebLoader
- experiment with ChromaDB vs FAISS
- store the chat history? 
- store the document indexes? 
- Chatbot over a SQL database https://python.langchain.com/docs/use_cases/sql/
