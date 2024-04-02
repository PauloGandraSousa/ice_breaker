#
# RAG from a PDF
#
# https://python.langchain.com/docs/get_started/quickstart
#

from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader

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


def execute(question):
    # execute the chain using RAG
    response = retrieval_chain.invoke({"input": question})
    answer = response["answer"]
    # print out answer
    print_out("RAG", question, answer)


def setup():
    global faiss_index, retrieval_chain

    # LLM
    llm = ChatOpenAI()
    # llm.invoke(question)
    #
    # RAG
    #
    # load additional documents using a web loader
    loader = PyPDFLoader("docs/eCafeteria-RFP.pdf")
    pages = loader.load_and_split()
    # use a vector store and embeddings
    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    # set up the chain that takes a question and the retrieved documents and generates an answer
    prompt = ChatPromptTemplate.from_template("""Assume you are a world class Business analyst. Your main goal is to understand the RFP and construct the backlog for the project. 
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    # However, we want the documents to first come from the retriever we just set up.
    # That way, we can use the retriever to dynamically select the most relevant documents and pass those in for a given question.
    retriever = faiss_index.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)


#
# main
#
if __name__ == "__main__":
    print("Hello LangChain!")
    # print(os.getenv("TITLE"))

    # setup the chain
    setup()

    # query the document
    execute("Please summarize the RFP.")
    execute("Please identity the different users of the system.")
    execute("Please identity the user stories for the kitchen manager.")

    # similarity search
    docs = faiss_index.similarity_search("How will the cafeteria user use the system?", k=2)
    for doc in docs:
        print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
