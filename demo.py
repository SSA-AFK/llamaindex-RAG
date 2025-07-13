import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

EMBEDDING_DIM = 1536#向量维度
COLLECTION_NAME = "full_demo"#集合名称
PATH = "./db"

client = QdrantClient(path=PATH)


Settings.llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX,api_key=os.getenv("DASHSCOPE_API_KEY"))
Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1)


Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=200)]

#加载
documents = SimpleDirectoryReader("./data").load_data()

if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)
#如果collection不存在则创建，存在则删除

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
)


vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)


storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

reranker = LLMRerank(top_n=2)
# 最终打分低于0.6的文档被过滤掉
sp = SimilarityPostprocessor(similarity_cutoff=0.6)#这是基于相似度打分

fusion_retriever = QueryFusionRetriever(
    [index.as_retriever()],
    similarity_top_k=5, # 检索召回 top k 结果
    num_queries=3,  # 生成 query 数
    use_async=False,
    # query_gen_prompt="",  # 可以自定义 query 生成的 prompt 模板
)

query_engine = RetrieverQueryEngine.from_args(
    fusion_retriever,
    node_postprocessors=[reranker],
    response_synthesizer=get_response_synthesizer(
        response_mode = ResponseMode.REFINE
    )
)

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine, 
    # condense_question_prompt="" # 可以自定义 chat message prompt 模板
)

while True:
    question=input("User:")
    if question.strip() == "":
        break
    response = chat_engine.chat(question)
    print(f"AI: {response}")