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
7. interactive chatbot - the user can enter questions and receive the answer
8. add loading multiple documents in different file formats. allows to define which folder to load, which system prompt to use as context and which set of predefined questions to use

Running the script, e.g.
  
    tutorial08-interactive-chat-RAG-multiple-filetypes.py docs/examples/DDD docs/questions/ddd.txt docs/prompts/ddd.txt 

## To do

- tools
- agents
- Doctran
- RetrievalQA
