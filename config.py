import os
from datetime import datetime
from dataclasses import dataclass, field


def get_current_date() -> str:
    """返回当前日期字符串，注入到所有 System Prompt 中"""
    return datetime.now().strftime("%Y-%m-%d")


def get_current_year() -> str:
    return datetime.now().strftime("%Y")


@dataclass
class DeepSeekConfig:
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    fast_model: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: int = 4096
    max_concurrent_calls: int = 5


@dataclass
class RAGConfig:
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 100
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 8
    faiss_index_path: str = "./data/faiss_index"
    knowledge_base_dir: str = ""    # 知识库目录，启动时自动扫描索引
    supported_extensions: tuple = (".txt", ".md", ".pdf")
    use_gpu: bool = False
    nprobe: int = 10


@dataclass
class MemoryConfig:
    max_short_term_turns: int = 20
    summary_threshold: int = 10
    long_term_index_path: str = "./data/long_term_memory"
    max_long_term_results: int = 5
    memory_decay_days: int = 30
    retrieval_min_score: float = 0.3
    retrieval_type_limits: dict = field(default_factory=lambda: {
        "preference": 2,
        "constraint": 2,
        "research_summary": 2,
        "episode": 1,
        "hypothesis": 1,
        "conflict": 0,
    })


@dataclass
class AgentConfig:
    max_planning_steps: int = 8
    max_reflection_rounds: int = 3
    confidence_threshold: float = 0.7
    max_context_tokens: int = 12000
    enable_trace_persistence: bool = True
    trace_db_path: str = "./data/agent_traces.sqlite3"


@dataclass
class AppConfig:
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data_dir: str = "./data"
    log_level: str = "INFO"


config = AppConfig()
