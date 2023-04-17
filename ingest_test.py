import pandas as pd
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis as RedisVectorStore

# set your openAI api key as an environment variable
os.environ['OPENAI_API_KEY'] = "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"
MAX_TEXT_LENGTH = 512
NUMBER_PRODUCTS = 2500  # Num products to use (subset)


def auto_truncate(val):
    """Truncate the given text."""
    return val[:MAX_TEXT_LENGTH]


def prepare_dataset():
    # Load Product data and truncate long text fields
    all_prods_df = pd.read_csv("product_data.csv", nrows=NUMBER_PRODUCTS, converters={
        'bullet_point': auto_truncate,
        'item_keywords': auto_truncate,
        'item_name': auto_truncate
    })

    # Contruct a primary key from item ID and domain name
    all_prods_df['primary_key'] = (
            all_prods_df['item_id'] + '-' + all_prods_df['domain_name']
    )
    # Replace empty strings with None and drop
    all_prods_df['item_keywords'].replace('', None, inplace=True)
    all_prods_df.dropna(subset=['item_keywords'], inplace=True)

    # Reset pandas dataframe index
    all_prods_df.reset_index(drop=True, inplace=True)

    print(all_prods_df.head())

    # Get the first 2500 products
    product_metadata = (
        all_prods_df
        .head(NUMBER_PRODUCTS)
        .to_dict(orient='index')
    )

    # Check one of the products
    print(product_metadata[0])
    return product_metadata


def setup_vector_store(product_metadata):
    # data that will be embedded and converted to vectors
    texts = [
        v['item_name'] for k, v in product_metadata.items()
    ]

    # product metadata that we'll store along our vectors
    metadatas = list(product_metadata.values())

    print("text: \n", texts[0], "\n metadatas\n", metadatas[0])

    # we will use OpenAI as our embeddings provider
    embedding = OpenAIEmbeddings()

    # name of the Redis search index to create
    index_name = "amazon_products"

    # assumes you have a redis stack server running on local host
    redis_url = "redis://localhost:6379"

    # create and load redis with documents
    vectorstore = RedisVectorStore.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        index_name=index_name,
        redis_url=redis_url
    )


if __name__ == "__main__":
    product_metadata = prepare_dataset()

    setup_vector_store(product_metadata)
