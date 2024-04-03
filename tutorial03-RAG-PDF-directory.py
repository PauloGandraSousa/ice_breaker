#
# RAG from a directory with several PDF files
#
# https://python.langchain.com/docs/get_started/quickstart
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
#

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()


class BusinessAnalystRagPdf:
    def __init__(self, title, directory):
        self.title = title
        self.directory = directory
        self.initialized = False
        self.retrieval_chain = None

    def print_out(self, question, answer):
        print(f"\n=== {self.title} ===")
        print(question)
        print(">>>>>>")
        print(answer)
        print("<<<<<<\n")

    def execute_query(self, question):
        if not self.initialized:
            self.set_up_rag_chain()

        # execute the chain using RAG
        response = self.retrieval_chain.invoke({"input": question})
        answer = response["answer"]
        # print out answer
        self.print_out(question, answer)

    def set_up_rag_chain(self):
        # LLM
        llm = ChatOpenAI()
        # llm.invoke(question)

        # load additional documents from a directory using a PDF loader
        loader = PyPDFDirectoryLoader(self.directory)
        pages = loader.load()
        # use a vector store and embeddings
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_documents(pages, embeddings)
        # set up the chain that takes a question and the retrieved documents and generates an answer
        prompt = ChatPromptTemplate.from_template("""Assume you are a world class Business analyst that works for a 
        software company building actuarial software. Your customers are insurance companies. You have received a set of 
        documents containing insurance product descriptions and rules. Your main goal is to understand those documents 
        and construct the backlog for the project.  
        Answer the following question based only on the provided context:
    
        <context>
        {context}
        </context>
    
        Question: {input}""")
        document_chain = create_stuff_documents_chain(llm, prompt)
        # However, we want the documents to first come from the retriever we just set up. That way, we can use the
        # retriever to dynamically select the most relevant documents and pass those in for a given question.
        retriever = faiss_index.as_retriever()
        self.retrieval_chain = create_retrieval_chain(retriever, document_chain)


#
# main
#
if __name__ == "__main__":
    print("LangChain! : RAG from a directory of PDF files")
    # print(os.getenv("TITLE"))

    # set up the chain
    rag = BusinessAnalystRagPdf("BA (RAG)", "docs/examples/MM/motor/")

    # query the document
    rag.execute_query("Please summarize the insurance product.")
    rag.execute_query("Please identity the mandatory and optional coverage of the product.")
    rag.execute_query("Please identity the different coverage packages.")
    rag.execute_query("Please identity the capital limits of each coverage.")
