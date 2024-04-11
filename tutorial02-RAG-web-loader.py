#
# LangChain tutorial
# step 2: RAG (Retrieval Augmented Generation) using a web loader
#
# https://python.langchain.com/docs/get_started/quickstart
#

from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()


#
# print out answer
#
def print_out(title, question, answer):
    print(f"\n=== {title} ===")
    print(question)
    print(">>>>>>")
    print(answer)
    print("<<<<<<\n")


#
# main
#
if __name__ == "__main__":
    print("Hello LangChain! : RAG from a web page")
    # print(os.getenv("TITLE"))

    question = "how can langsmith help with testing?"

    # LLM
    llm = ChatOpenAI()

    # load additional documents using a web loader
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()

    # prepare the documents
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)

    # use a vector store and embeddings
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)

    # set up the chain that takes a question and the retrieved documents and generates an answer
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    # However, we want the documents to first come from the retriever we just set up.
    # That way, we can use the retriever to dynamically select the most relevant documents and pass those in for a given question.
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # execute the chain using RAG
    response = retrieval_chain.invoke({"input": question})
    answer = response["answer"]

    # print out answer
    print_out("RAG", question, answer)
