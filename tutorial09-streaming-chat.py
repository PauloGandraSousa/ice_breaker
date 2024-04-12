#
# LangChain tutorial
# step 9: Conversational Bot with minimal interactive UI that streams the answers from the LLM
#         with chat history and RAG
#         RAG from a directory with several files of different formats
#         folders, set of standard questions and system prompt can be entered as runtime program arguments
#
# Running the script, e.g.
#
#     tutorial09-streaming-chat.py docs/examples/DDD docs/questions/ddd.txt docs/prompts/ddd.txt
#
#
# Based on the following tutorials:
#
# https://python.langchain.com/docs/get_started/quickstart
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
# https://python.langchain.com/docs/integrations/document_loaders/microsoft_excel/
# https://python.langchain.com/docs/use_cases/question_answering/chat_history
# https://betterprogramming.pub/building-a-multi-document-reader-and-chatbot-with-langchain-and-chatgpt-d1864d47e339
# https://python.langchain.com/docs/use_cases/question_answering/streaming/
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
import string
import sys
from dotenv import load_dotenv
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader


load_dotenv()


class ChatUI:
    def __init__(self, title: string, bot, standard_questions: list[string]):
        self.title = title
        self.standard_questions = standard_questions
        self.bot = bot
        self.__count = 0
        print(f"\n=== {self.title} ===\n")

    def __answer_and_print(self, question: string):
        print(f"A {self.__count}:")
        bot.stream(question, lambda msg: print(msg, end=""))
        print()

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

    def __read_question(self) -> string:
        self.__count = self.__count + 1
        self.print_frame()
        question = input(f"Q {self.__count}: ")
        return question

    def ask(self, question: string):
        self.__count = self.__count + 1
        self.print_frame()
        print(f"Q {self.__count}. {question}")
        self.__answer_and_print(question)

    def ask_standard_questions(self):
        print("\n<standard-questions>\n")
        [self.ask(q) for q in self.standard_questions]
        print("\n</standard-questions>\n")


class ConversationalBusinessAnalystRag:
    def __init__(self, system_context: string, directory: string, load_now=False):
        self.__directory = directory
        if "{context}" not in system_context:
            self.__system_context = (
                "You are an assistant for question-answering tasks.\n"
                + system_context
                + "\n\nContext: {context}"
            )
        else:
            self.__system_context = system_context
        self.__retrieval_chain = None
        self.chat_history = []
        if load_now:
            self.__set_up_rag_chain()

    def chat(self, question):
        """executes synchronous invoke of the chain to answer a question and returns the answer from the chain"""
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

    def stream(self, question, printer_callback):
        """streams the results of the chain when answering a question"""
        if not self.__is_initialized():
            self.__set_up_rag_chain()

        # execute the chain using RAG and stream the results
        response_stream = self.__retrieval_chain.stream(
            {"chat_history": self.chat_history, "input": question}
        )
        # streaming
        answer = self.__stream_answer(response_stream, printer_callback)

        # update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer

    def __stream_answer(self, response_stream, printer_callback):
        output = {}
        curr_key = None
        for chunk in response_stream:
            for key in chunk:
                if key not in output:
                    output[key] = chunk[key]
                else:
                    output[key] += chunk[key]
                '''
                if key != curr_key:
                    print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
                else:
                    print(chunk[key], end="", flush=True)
                '''
                if key == "answer":
                    printer_callback(chunk[key])
                curr_key = key
        answer = output["answer"]
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
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.__system_context),
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
        llm = ChatOpenAI(temperature=0)  # TODO ChatOpenAI(model="gpt-4", temperature=0)
        # load documents for RAG
        pages = self.__load_documents()
        # set up retriever
        retriever_chain = self.__set_up_retriever(llm, pages)
        # set up retrieval chain
        self.__retrieval_chain = self.__set_up_retrieval_chain(llm, retriever_chain)

    def __is_initialized(self) -> bool:
        return self.__retrieval_chain is not None

    def __load_documents_pdf_directory(self) -> list[Document]:
        # load documents from a directory using a PDF loader
        loader = PyPDFDirectoryLoader(self.__directory)
        pages = loader.load()
        return pages

    def __load_documents(self) -> list[Document]:
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
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_documents = text_splitter.split_documents(documents)

        return chunked_documents

    def __build_loader(self, file:string) -> None | BaseLoader:
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
def read_standard_questions() -> list[string]:
    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        filename = input(
            "Which file contains the standard questions to query the documents (e.g., docs/questions/ddd.txt)?"
        )
    f = open(filename, "r")
    return f.readlines()


def read_folder() -> string:
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return input("Which directory would you like to load files from (e.g., docs/examples/DDD)?")


def read_system_prompt() -> string:
    if len(sys.argv) > 3:
        f = open(sys.argv[3], "r")
        prompt = f.read()
    else:
        prompt = """You are an assistant for question-answering tasks. Use
        the following pieces of retrieved context to answer the question. If you don't know the answer, just say that 
        you don't know.
        
        
        Context: {context}"""
    return prompt


if __name__ == "__main__":
    print("LangChain! : ChatBot with RAG\n")
    # print(os.getenv("TITLE"))

    # input "parameters"
    folder = read_folder()
    standard_questions = read_standard_questions()
    system_prompt = read_system_prompt()

    # set up the chain
    bot = ConversationalBusinessAnalystRag(system_prompt, folder, load_now=True)

    # minimal UI
    ui = ChatUI("BA (RAG)", bot, standard_questions)

    # query the document
    ui.interactive()
