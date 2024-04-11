#
# Conversational RAG from a directory with several PDF files
#
# https://python.langchain.com/docs/get_started/quickstart
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
# https://python.langchain.com/docs/use_cases/question_answering/chat_history
#

from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()


class ChatUI:
    def __init__(self, title, bot):
        self.title = title
        self.bot = bot
        self.count = 0
        print(f"\n=== {self.title} ===\n")

    def __answer_and_print(self, question):
        answer = bot.chat(question)
        self.__ai_prompt(answer)

    def __ai_prompt(self, answer):
        print(f"A {self.count}: >>>")
        print(answer)
        print(f"<<< (A {self.count})\n")

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

    def __read_question(self):
        self.count = self.count + 1
        question = input(f"Q {self.count}: ")
        return question

    def ask(self, question):
        self.count = self.count + 1
        print(f"Q {self.count}. {question}")
        self.__answer_and_print(question)

    def ask_standard_questions(self):
        standard_questions = [
            "Please summarize the insurance product.",
            "what type of risks are covered?",
            "Please tell me the name of all the different coverage packages.",
            "Please tell me the name of all the different covers and to which coverage package they belong.",
            "Which are the mandatory and optional coverage of the package?",
            "Can I insure my Cessna Citation Latitude?",
        ]
        print("-- start of standard questions --")
        [self.ask(q) for q in standard_questions]
        print("-- end of standard questions --")


class ConversationalBusinessAnalystRagPdf:
    def __init__(self, directory):
        self.__directory = directory
        self.__initialized = False
        self.__retrieval_chain = None
        self.chat_history = []

    def chat(self, question):
        if not self.__initialized:
            self.__set_up_rag_chain()
            self.__initialized = True

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
        understand those documents and construct the backlog for the project. It is important to understand the 
        different coverage packages the company wants to have and if the coverage is mandatory or not in that package. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say 
        that you don't know. 


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

    def __load_documents(self):
        # load additional documents from a directory using a PDF loader
        loader = PyPDFDirectoryLoader(self.__directory)
        pages = loader.load()
        # TODO load other documents in other formats
        return pages


#
# main
#
if __name__ == "__main__":
    print("LangChain! : ChatBot RAG from a directory of PDF files")
    # print(os.getenv("TITLE"))

    # set up the chain
    bot = ConversationalBusinessAnalystRagPdf("docs/examples/MM/motor/")

    # minimal UI
    ui = ChatUI("BA (RAG)", bot)

    # query the document
    ui.interactive()
