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


class RagPdf:
    def __init__(self, title):
        self.title = title
        self.retrieval_chain = None

    def print_out(self, question, answer):
        print(f"\n=== {self.title} ===")
        print(question)
        print(">>>>>>")
        print(answer)
        print("<<<<<<\n")

    def execute_query(self, question):
        # execute the chain using RAG
        response = self.retrieval_chain.invoke({"input": question})
        answer = response["answer"]
        # print out answer
        self.print_out(question, answer)

    def set_up_rag_chain(self, filename):
        # LLM
        llm = ChatOpenAI()
        # llm.invoke(question)

        # load additional documents using a web loader
        loader = PyPDFLoader(filename)
        pages = loader.load_and_split()
        # use a vector store and embeddings
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_documents(pages, embeddings)
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
        self.retrieval_chain = create_retrieval_chain(retriever, document_chain)


#
# main
#
if __name__ == "__main__":
    print("LangChain! : RAG from a PDF")
    # print(os.getenv("TITLE"))

    # set up the chain
    rag = RagPdf("RAG")
    rag.set_up_rag_chain("docs/eCafeteria-RFP.pdf")

    # query the document
    rag.execute_query("Please summarize the RFP.")
    rag.execute_query("Please identity the different users of the system.")
    rag.execute_query("Please identity the user stories for the kitchen manager.")
