#
# LangChain tutorial
# step 1: just invoke a LLM
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
    print("Hello LangChain!")
    # print(os.getenv("TITLE"))

    question = "how can langsmith help with testing?"

    # LLM
    llm = ChatOpenAI()
    # llm.invoke("how can langsmith help with testing?")

    # prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are world class technical documentation writer."),
            ("user", "{input}"),
        ]
    )

    # output parser
    output_parser = StrOutputParser()

    # set up the chain
    chain = prompt | llm | output_parser

    # execute the chain
    answer = chain.invoke({"input": question})

    # print out answer
    print_out("LLM", question, answer)
