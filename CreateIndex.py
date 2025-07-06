
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

# Your AWS region and domain endpoint
region = "ap-south-1"
host = "https://search-findomain1-lgyucsnynjo3aejlv5cmxnp64q.ap-south-1.es.amazonaws.com"

# Get AWS credentials
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "es",
    session_token=credentials.token
)

# Create OpenSearch client
client = OpenSearch(
    hosts=[{"host": host.replace("https://", ""), "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

index_name = "financialinfo"
dimension = 768  # Use the dimension of your embedding model

index_body = {
    "settings": {
        "index": {
            "knn": True  # Enable k-NN plugin for vector search
        }
    },
    "mappings": {
        "properties": {
            "content": { "type": "text" },
            "vector_field": {
                "type": "knn_vector",
                "dimension": dimension
            },
            "source": { "type": "keyword" }  # optional metadata field
        }
    }
}

# Create the index
if not client.indices.exists(index=index_name):
    response = client.indices.create(index=index_name, body=index_body)
    print("Index created:", response)
else:
    print("Index already exists.")
