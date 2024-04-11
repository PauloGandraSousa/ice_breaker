#
# Conversational RAG from a directory with several files of different formats
#
# https://python.langchain.com/docs/get_started/quickstart
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
# https://python.langchain.com/docs/integrations/document_loaders/microsoft_excel/
# https://python.langchain.com/docs/use_cases/question_answering/chat_history
# https://betterprogramming.pub/building-a-multi-document-reader-and-chatbot-with-langchain-and-chatgpt-d1864d47e339
#
#
# In order to use the 'unstructured' package a couple of system dependencies are needed,
# check the dependencies listed at https://pypi.org/project/unstructured/
# check also https://unstructured-io.github.io/unstructured/installation/full_installation.htm
# namely (it seems) you'll need to install the following on your machine:
#   - Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
#   - Tesseract OCR: https://github.com/tesseract-ocr/tesseract
#   - Pandoc: https://github.com/jgm/pandoc/releases/
#   - Poppler: https://github.com/oschwartz10612/poppler-windows/releases
#  you will need to add the location of the DLL files for these libraries into your PATH environment variable
#
import os

from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader

load_dotenv()


class ChatUI:
    def __init__(self, title, bot, standard_questions):
        self.title = title
        self.standard_questions = standard_questions
        self.bot = bot
        self.__count = 0
        print(f"\n=== {self.title} ===\n")

    def __answer_and_print(self, question):
        answer = bot.chat(question)
        self.__ai_prompt(answer)

    def __ai_prompt(self, answer):
        print(f"A {self.__count}:")
        print(answer)

    def interactive(self):
        print(
            "How may I assist you today? (enter 'quit' to end or 'standard' for the set of predefined questions)\n"
        )
        question = self.__read_question()
        while question != "end":
            if question == "standard":
                self.ask_standard_questions()
            else:
                self.__answer_and_print(question)
            question = self.__read_question()
        print("\nIt has been a pleasure helping you. Come back soon.")
        print("=== end of conversation ===")

    def print_frame(self):
        print("------------------------------------------------------\n")

    def __read_question(self):
        self.__count = self.__count + 1

        self.print_frame()
        question = input(f"Q {self.__count}: ")
        return question

    def ask(self, question):
        self.__count = self.__count + 1
        self.print_frame()
        print(f"Q {self.__count}. {question}")
        self.__answer_and_print(question)

    def ask_standard_questions(self):
        print("\n<standard-questions>\n")
        [self.ask(q) for q in self.standard_questions]
        print("\n</standard-questions>\n")


class ConversationalBusinessAnalystRag:
    def __init__(self, directory, load_now=False):
        self.__directory = directory
        self.__retrieval_chain = None
        self.chat_history = []
        if load_now:
            self.__set_up_rag_chain()

    def chat(self, question):
        if not self.__is_initialized():
            self.__set_up_rag_chain()

        # execute the chain using RAG
        response = self.__retrieval_chain.invoke(
            {"chat_history": self.chat_history, "input": question}
        )
        answer = response["answer"]

        # update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer

    def __set_up_retriever(self, llm, documents):
        # use a vector store and embeddings for the external documents
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(documents, embeddings)
        # create a retriever
        retriever = vector_store.as_retriever()

        # The retrieval method should take the whole history into account
        # First we need a prompt that we can pass into an LLM to generate this search query
        contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference 
        context in the chat history, generate a search query to look up to get information relevant to the 
        conversation."""
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                ("user", contextualize_q_system_prompt),
            ]
        )
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        return retriever_chain

    def __test_retriever(self, retriever_chain):
        # test the retriever chain
        chat_history = [
            HumanMessage(content="Does the product cover theft?"),
            AIMessage(content="Yes!"),
        ]
        res = retriever_chain.invoke(
            {"chat_history": chat_history, "input": "Tell me the damage limits"}
        )
        print(f"\nDEBUG: {res}\n")

    def __set_up_retrieval_chain(self, llm, retriever_chain):
        # Now that we have this new retriever, we can create a new chain to continue the conversation with these
        # retrieved documents in mind.
        system_prompt = """You are an assistant for question-answering tasks. Assume you are a world class Business 
        analyst that works for a software company building actuarial software. Your customers are insurance companies. 
        You have received a set of documents containing insurance product descriptions and rules. Your main goal is to 
        understand those documents and construct the backlog for the project. You need to understand what coverage the 
        product offers as well as the limits, exclusions, and co-payments for each coverage. It is also important to 
        understand the different coverage packages the company wants to have and if the coverage is mandatory or not in 
        that package. Another area you need to pay attention is the business rules for premium calculation - which 
        premium should the customer pay for each coverage and which tariffication tables to use for the calculation. Use
        the following pieces of retrieved context to answer the question. If you don't know the answer, just say that 
        you don't know. 


        Context: {context}"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(llm, prompt)

        # set up the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        return retrieval_chain

    def __set_up_rag_chain(self):
        # LLM
        llm = ChatOpenAI()
        # load documents for RAG
        pages = self.__load_documents()
        # set up retriever
        retriever_chain = self.__set_up_retriever(llm, pages)
        # set up retrieval chain
        self.__retrieval_chain = self.__set_up_retrieval_chain(llm, retriever_chain)

    def __is_initialized(self):
        return self.__retrieval_chain != None

    def __load_documents_pdf_directory(self):
        # load documents from a directory using a PDF loader
        loader = PyPDFDirectoryLoader(self.__directory)
        pages = loader.load()
        return pages

    def __load_documents(self):
        # load additional documents from a directory using the appropriate loader
        documents = []
        for file in os.listdir(self.__directory):
            loader = self.__build_loader(file)
            if loader:
                documents.extend(loader.load())
                # debug
                print(f"INFO: loaded document {file}")
            else:
                # debug
                print(f"WARN: ignored document {file}")

        # we split the data into chunks of 1,000 characters, with an overlap of 200 characters between the chunks,
        # which helps to give better results and contain the context of the information between chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents(documents)

        return chunked_documents

    def __build_loader(self, file):
        # we might need to check the performance and accuracy of each loader. The 'unstructured' package is able to load
        # all of these file types, so we might use just the 'UnstructuredFileLoader' loader for all file types.
        doc_path = os.path.join(self.__directory, file)
        loader = None
        if file.endswith(".pdf"):
            loader = PyPDFLoader(doc_path)
        elif file.endswith(".xlsx") or file.endswith(".xls"):
            loader = UnstructuredExcelLoader(doc_path, mode="elements")
        elif file.endswith(".csv"):
            loader = CSVLoader(file_path=doc_path)
            """
            loader = CSVLoader(
                file_path="./example_data/mlb_teams_2012.csv",
                csv_args={
                    "delimiter": ",",
                    "quotechar": '"',
                    "fieldnames": ["MLB Team", "Payroll in millions", "Wins"],
                },
            )
            """
        elif file.endswith(".pptx") or file.endswith(".ppt"):
            loader = UnstructuredPowerPointLoader(doc_path, mode="elements")
        elif file.endswith(".docx") or file.endswith(".doc"):
            loader = Docx2txtLoader(doc_path)
        elif file.endswith(".txt"):
            loader = TextLoader(doc_path)
        return loader


#
# main
#
def read_standard_questions():
    print(
        "Which file contains the standard questions to query the documents (e.g., docs/questions/ddd.txt)?"
    )
    filename = input("(enter for default)")
    if filename == "":
        filename = "docs/questions/insurance_BA.txt"
    f = open(filename, "r")
    return f.readlines()


def read_folder():
    print(
        "Which directory would you like to load files from (e.g., docs/examples/DDD)?"
    )
    folder = input("(enter for default)")
    if folder == "":
        folder = "docs/examples/MM/motor"
    return folder


if __name__ == "__main__":
    print("LangChain! : ChatBot RAG from a directory of files")
    # print(os.getenv("TITLE"))

    # input "parameters"
    folder = read_folder()
    standard_questions = read_standard_questions()

    # set up the chain
    bot = ConversationalBusinessAnalystRag(folder, load_now=True)

    # minimal UI
    ui = ChatUI("BA (RAG)", bot, standard_questions)

    # query the document
    ui.interactive()
