"""
目录索引器 — 批量扫描 + 增量索引 + 去重

支持:
- 扫描指定目录下的所有 .txt .md .pdf 文件（含子目录）
- 增量索引: 已索引的文件跳过（基于 文件名+修改时间 去重）
- 进度显示: 逐文件报告索引进度
- 错误容忍: 单个文件解析失败不阻塞其他文件
"""
import os
import json
import uuid
import logging
from typing import Optional, Callable
from config import config
from rag.chunker import RecursiveChunker, StructuredChunker, DocumentChunk
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# 索引清单文件: 记录哪些文件已被索引
MANIFEST_FILENAME = "index_manifest.json"


class DirectoryIndexer:
    """
    目录批量索引器

    去重策略:
    - 维护一个 manifest 文件 (index_manifest.json)，记录已索引文件的路径和修改时间
    - 扫描时比对: 路径+mtime 都匹配 → 跳过; mtime 变了 → 重新索引
    - manifest 和 FAISS 索引一起存储在 data_dir 下
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.manifest_path = os.path.join(config.data_dir, MANIFEST_FILENAME)
        self.manifest: dict[str, dict] = self._load_manifest()

    def _load_manifest(self) -> dict:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        os.makedirs(os.path.dirname(self.manifest_path) if os.path.dirname(self.manifest_path) else ".", exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def scan_directory(self, directory: str) -> dict:
        """
        扫描目录，返回需要索引的文件列表

        Returns:
            {
                "new_files": [...],       # 从未索引过
                "updated_files": [...],   # 修改过需重新索引
                "skipped_files": [...],   # 未修改，跳过
                "unsupported_files": [...], # 不支持的格式
            }
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")

        new_files = []
        updated_files = []
        skipped_files = []
        unsupported_files = []

        for root, dirs, files in os.walk(directory):
            # 跳过隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                filepath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()

                if ext not in config.rag.supported_extensions:
                    unsupported_files.append(filepath)
                    continue

                mtime = os.path.getmtime(filepath)
                abs_path = os.path.abspath(filepath)

                if abs_path in self.manifest:
                    if self.manifest[abs_path]["mtime"] == mtime:
                        skipped_files.append(filepath)
                    else:
                        updated_files.append(filepath)
                else:
                    new_files.append(filepath)

        return {
            "new_files": new_files,
            "updated_files": updated_files,
            "skipped_files": skipped_files,
            "unsupported_files": unsupported_files,
        }

    def index_directory(
        self,
        directory: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> dict:
        """
        索引目录下所有支持的文件

        Args:
            directory: 目录路径
            progress_callback: 进度回调 fn(filename, current, total)

        Returns:
            {"indexed": int, "skipped": int, "failed": int, "errors": [...]}
        """
        scan = self.scan_directory(directory)
        to_index = scan["new_files"] + scan["updated_files"]

        if not to_index:
            return {
                "indexed": 0,
                "skipped": len(scan["skipped_files"]),
                "failed": 0,
                "errors": [],
                "total_files": len(scan["skipped_files"]),
            }

        all_chunks: list[DocumentChunk] = []
        indexed = 0
        failed = 0
        errors = []

        for i, filepath in enumerate(to_index):
            filename = os.path.basename(filepath)
            if progress_callback:
                progress_callback(filename, i + 1, len(to_index))

            try:
                chunks = self._index_single_file(filepath)
                if chunks:
                    all_chunks.extend(chunks)
                    indexed += 1
                    # 更新 manifest
                    abs_path = os.path.abspath(filepath)
                    self.manifest[abs_path] = {
                        "mtime": os.path.getmtime(filepath),
                        "filename": filename,
                        "num_chunks": len(chunks),
                        "doc_id": chunks[0].doc_id if chunks else "",
                    }
                else:
                    failed += 1
                    errors.append(f"{filename}: 内容为空")
            except Exception as e:
                failed += 1
                errors.append(f"{filename}: {str(e)}")
                logger.error(f"Failed to index {filepath}: {e}")

        # 批量建立/更新索引
        if all_chunks:
            if self.retriever.faiss_index is None or self.retriever.faiss_index.ntotal == 0:
                self.retriever.build_index(all_chunks)
            else:
                self._add_chunks_batch(all_chunks)

            self.retriever.save_index(config.rag.faiss_index_path)

        self._save_manifest()

        return {
            "indexed": indexed,
            "skipped": len(scan["skipped_files"]),
            "failed": failed,
            "errors": errors,
            "total_files": indexed + len(scan["skipped_files"]) + failed,
        }

    def _index_single_file(self, filepath: str) -> list[DocumentChunk]:
        """解析并分块单个文件"""
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()

        # 读取文件内容
        text = ""
        if ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(filepath)
                text = "\n\n".join([page.get_text() for page in doc])
                doc.close()
            except ImportError:
                raise RuntimeError("需要 PyMuPDF: pip install PyMuPDF")

        if not text.strip():
            return []

        # 选择分块策略
        doc_id = str(uuid.uuid4())[:8]
        if ext == ".md":
            chunker = StructuredChunker(
                chunk_size=config.rag.chunk_size,
                chunk_overlap=config.rag.chunk_overlap,
            )
        else:
            chunker = RecursiveChunker(
                chunk_size=config.rag.chunk_size,
                chunk_overlap=config.rag.chunk_overlap,
                min_chunk_size=config.rag.min_chunk_size,
            )

        return chunker.chunk_document(
            text=text,
            doc_id=doc_id,
            doc_title=filename,
            metadata={
                "source": "directory",
                "filename": filename,
                "filepath": os.path.abspath(filepath),
            },
        )

    def _add_chunks_batch(self, chunks: list[DocumentChunk]):
        """批量增量添加 chunks 到已有索引"""
        texts = [c.content for c in chunks]
        embeddings = self.retriever.get_embeddings(texts)
        self.retriever.faiss_index.add(embeddings)
        self.retriever.chunks.extend(chunks)
        self.retriever.bm25_index.add_documents(chunks)

    def get_manifest_stats(self) -> dict:
        """获取已索引文件统计"""
        return {
            "indexed_files": len(self.manifest),
            "files": [
                {"path": p, "filename": info["filename"], "chunks": info["num_chunks"]}
                for p, info in self.manifest.items()
            ],
        }

    def clear_index(self):
        """清空索引和 manifest"""
        self.manifest = {}
        self._save_manifest()
        import faiss
        self.retriever.faiss_index = faiss.IndexFlatIP(self.retriever.embedding_dim)
        self.retriever.chunks = []
        self.retriever.bm25_index = type(self.retriever.bm25_index)()
        logger.info("Index and manifest cleared")
