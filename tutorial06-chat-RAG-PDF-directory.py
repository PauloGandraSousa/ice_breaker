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

    def print_out(self, question, answer):
        print(">>>>>>")
        print(f"Q {self.count}. {question}")
        print("------")
        print(answer)
        print("<<<<<<\n")

    def ask(self, question):
        answer = bot.chat(question)
        self.count = self.count + 1
        self.print_out(question, answer)


class ConversationalBusinessAnalystRagPdf:
    def __init__(self, directory):
        self.directory = directory
        self.initialized = False
        self.retrieval_chain = None
        self.chat_history = []

    def chat(self, question):
        if not self.initialized:
            self.set_up_rag_chain()
            self.initialized = True

        # execute the chain using RAG
        response = self.retrieval_chain.invoke(
            {"chat_history": self.chat_history, "input": question}
        )
        answer = response["answer"]

        # update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer

    def set_up_retriever(self, llm, documents):
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

    def test_retriever(self, retriever_chain):
        # test the retriever chain
        chat_history = [
            HumanMessage(content="Does the product cover theft?"),
            AIMessage(content="Yes!"),
        ]
        res = retriever_chain.invoke(
            {"chat_history": chat_history, "input": "Tell me the damage limits"}
        )
        print(f"\nDEBUG: {res}\n")

    def set_up_retrieval_chain(self, llm, retriever_chain):
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

    def set_up_rag_chain(self):
        # LLM
        llm = ChatOpenAI()

        # load additional documents from a directory using a PDF loader
        loader = PyPDFDirectoryLoader(self.directory)
        pages = loader.load()

        # set up retriever
        retriever_chain = self.set_up_retriever(llm, pages)

        # set up retrieval chain
        self.retrieval_chain = self.set_up_retrieval_chain(llm, retriever_chain)


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
    ui.ask("Please summarize the insurance product.")
    ui.ask("Which are the mandatory and optional coverage of the product?")
    ui.ask("What are the exclusions of the first coverage?")
    ui.ask("Please tell me more about the third coverage you mentioned above.")
    ui.ask("Can I insure my Cesena 900?")
    ui.ask("Please identity all the different coverage packages.")
    ui.ask("Please identity the capital limits of each coverage.")
