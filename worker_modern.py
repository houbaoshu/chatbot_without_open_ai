import os
from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
    ChatHuggingFace,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 导入构建新版问答链的关键函数
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 全局变量
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None


def init_llm():
    global llm_hub, embeddings

    # 从环境变量获取 Token
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        # 如果没有设置环境变量，请在这里手动填写或确保环境中有此变量
        # hf_token = "你的 HF TOKEN"
        raise ValueError("请设置 HUGGINGFACEHUB_API_TOKEN 环境变量")

    repo_id = "Qwen/Qwen2.5-72B-Instruct"  # 更换为支持对话任务且中文能力更强的模型

    # 1. 初始化模型 (LLM)
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
    )
    llm_hub = ChatHuggingFace(llm=llm_endpoint)

    # 2. 初始化 Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )


def process_document(document_path):
    global conversation_retrieval_chain

    # 1. 加载文档
    loader = PyPDFLoader(document_path)
    documents = loader.load()

    # 2. 文档切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    # 3. 创建向量检索库 (Chroma) 并转为检索器 (Retriever)
    db = Chroma.from_documents(texts, embedding=embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    # 4. 定义 Prompt 模板
    # 系统提示词定义了 AI 的角色和如何使用背景知识
    system_prompt = (
        "你是一个专门负责回答文档相关问题的助教。"
        "请根据以下检索到的上下文（context）来回答问题（input）。"
        "如果你不知道答案，就说不知道，不要胡编乱造。"
        "答案请保持简洁、专业。"
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. 使用 LCEL (LangChain Expression Language) 构建文档处理和问答链
    conversation_retrieval_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm_hub
        | StrOutputParser()
    )


def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    if conversation_retrieval_chain is None:
        return "请先上传一个 PDF 文档。"

    # 7. 调用 LCEL 问答链
    # 返回的结果直接是解析好的字符串答案
    answer = conversation_retrieval_chain.invoke(prompt)

    # 更新对话历史 (可选，用于保持状态)
    chat_history.append((prompt, answer))

    return answer


# 执行初始化
init_llm()
