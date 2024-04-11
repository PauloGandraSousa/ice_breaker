#
# LangChain tutorial
# step 3: Conversational Bot with chat history and RAG
#
# https://python.langchain.com/docs/get_started/quickstart
#

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

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


def load_documents():
    # load additional documents using a web loader
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()

    # prepare the documents
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    return documents


def test_retriever(retriever_chain):
    # test the retriever chain
    chat_history = [
        HumanMessage(content="Can LangSmith help test my LLM applications?"),
        AIMessage(content="Yes!"),
    ]
    retriever_chain.invoke({"chat_history": chat_history, "input": "Tell me how"})


def set_up_retriever():
    # use a vector store and embeddings for the external documents
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(load_documents(), embeddings)
    # create a retriever
    retriever = vector.as_retriever()

    # The retrieval method should take the whole history into account
    # First we need a prompt that we can pass into an LLM to generate this search query
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def set_up_retrieval_chain(retriever_chain):
    # Now that we have this new retriever, we can create a new chain to continue the conversation with these retrieved
    # documents in mind.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    # set up the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    return retrieval_chain


#
# main
#
if __name__ == "__main__":
    print("Hello LangChain! : RAG from a web page")
    # print(os.getenv("TITLE"))

    # set up LLM
    llm = ChatOpenAI()
    # set up retriever
    retriever_chain = set_up_retriever()
    # debug only! >>>
    test_retriever(retriever_chain)
    # <<<<

    # set up retrieval chain
    retrieval_chain = set_up_retrieval_chain(retriever_chain)

    #
    # execute the chain using RAG
    #

    # first question
    question = "Can LangSmith help test my LLM applications?"
    chat_history = []
    response = retrieval_chain.invoke({"chat_history": chat_history, "input": question})
    answer = response["answer"]
    print_out("RAG", question, answer)

    # followup question
    followup_question = "Tell me how"
    chat_history = [HumanMessage(content=question), AIMessage(content=answer)]
    response = retrieval_chain.invoke(
        {"chat_history": chat_history, "input": followup_question}
    )
    answer = response["answer"]
    print_out("RAG", followup_question, answer)
