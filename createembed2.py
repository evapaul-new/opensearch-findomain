from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from requests_aws4auth import AWS4Auth
import boto3

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return chunks

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

region = "ap-south-1"
host = "search-findomain1-lgyucsnynjo3aejlv5cmxnp64q.ap-south-1.es.amazonaws.com"

# Get AWS credentials (frozen)
credentials = boto3.Session().get_credentials().get_frozen_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "es",
    session_token=credentials.token
)

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Ensure index exists
client.indices.create(index="financialinfo", ignore=400)

vectorstore = OpenSearchVectorSearch(
    index_name="financialinfo",
    opensearch_url=f"https://{host}",
    opensearch_client=client,
    embedding_function=embedding_function
)

docs = load_and_split_pdf("/Users/evangeline/ML_Projects/OpenSearch/rag_data/Personal Finance for Dummies.pdf")

from math import ceil

batch_size = 500
for i in range(ceil(len(docs) / batch_size)):
    batch = docs[i * batch_size : (i + 1) * batch_size]
    vectorstore.add_documents(batch)

#vectorstore.add_documents(docs)
#print("Printing vectorstore details:")
#print(vars(vectorstore))