from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.redis import Redis

from MyRedisProductRetriever import MyRedisProductRetriever
from prompt import condense_question_prompt, qa_prompt
import os

os.environ['OPENAI_API_KEY'] = "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"

embeddings = OpenAIEmbeddings()


def db_test(rds):
    query = "Can you provide more information about the Bourge Men's Moda-32 Sea Green Running Shoes in size 8 UK (42 " \
            "EU) (9 US) (Moda-32-08)?"
    results = rds.similarity_search(query)
    print(results[0].page_content)


def get_chain(retriever):
    # define two LLM models from OpenAI
    llm = ChatOpenAI(temperature=0)

    streaming_llm = ChatOpenAI(
        streaming=True,
        callback_manager=CallbackManager([
            StreamingStdOutCallbackHandler()
        ]),
        verbose=True,
        max_tokens=150,
        temperature=0
    )

    # use the LLM Chain to create a question creation chain
    question_generator = LLMChain(
        llm=llm,
        prompt=condense_question_prompt
    )

    # use the streaming LLM to create a question answering chain
    doc_chain = load_qa_chain(
        llm=streaming_llm,
        chain_type="stuff",
        prompt=qa_prompt
    )

    return ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
        question_generator=question_generator
    )


if __name__ == "__main__":
    # Load from existing index
    rds = Redis.from_existing_index(embeddings, redis_url="redis://localhost:6379", index_name='amazon_products')
    # redis_product_retriever = rds.as_retriever()
    redis_product_retriever = MyRedisProductRetriever(vectorstore=rds)

    # db_test(redis_product_retriever)

    chatbot = get_chain(redis_product_retriever)

    # create a chat history buffer
    chat_history = []
    # gather user input for the first question to kick off the bot
    question = input("Hi! What are you looking for today?")

    # keep the bot running in a loop to simulate a conversation
    while True:
        result = chatbot(
            {"question": question, "chat_history": chat_history}
        )
        # print(result["answer"])
        print("\n")
        chat_history.append((result["question"], result["answer"]))
        question = input()
