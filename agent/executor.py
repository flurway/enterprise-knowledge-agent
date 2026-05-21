"""
任务执行器 — 调用工具 + 管理中间状态
"""
import logging
import time
from models.deepseek import llm_client
from rag.retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker, CitationTracer
from config import config
from agent.policies import ToolPolicy, preview_payload
from agent.state import ToolCallRecord
from agent.hooks import HookContext, HookManager
from agent.mcp_adapter import McpToolAdapter
from agent.tool_dispatcher import ToolDispatcher, ToolDispatchError, ToolInvocationContext
from agent.tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry
from security.injection_detector import scan_text
from security.untrusted_flow import suspicious_dependency_ids

logger = logging.getLogger(__name__)


class StepResult:
    def __init__(self, step_id: int, action: str, success: bool,
                 output: str = "", data: dict = None, error: str = ""):
        self.step_id = step_id
        self.action = action
        self.success = success
        self.output = output
        self.data = data or {}
        self.error = error

    def to_dict(self) -> dict:
        return {"step_id": self.step_id, "action": self.action,
                "success": self.success, "output": self.output[:500], "error": self.error}


class TaskExecutor:
    def __init__(self, retriever: HybridRetriever, tool_policy: ToolPolicy | None = None,
                 hook_manager: HookManager | None = None,
                 tool_registry: ToolRegistry | None = None,
                 dispatcher: ToolDispatcher | None = None,
                 mcp_adapter: McpToolAdapter | None = None):
        self.retriever = retriever
        self.reranker = CrossEncoderReranker(use_model_rerank=False)
        self.citation_tracer = CitationTracer()
        self.execution_context: dict[int, StepResult] = {}
        self.all_citations: dict = {}
        self.tool_policy = tool_policy or ToolPolicy()
        self.tool_calls: list[ToolCallRecord] = []
        self.hook_manager = hook_manager or HookManager()
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY
        self.dispatcher = dispatcher or ToolDispatcher(self.tool_registry)
        self.mcp_adapter = mcp_adapter or McpToolAdapter(self.tool_registry)
        self._register_local_tools()
        self.mcp_adapter.register_with(self.dispatcher)

    async def execute_step(self, step: dict) -> StepResult:
        action = step.get("action", "")
        step_id = step.get("step_id", 0)
        params = step.get("input_params", {})
        logger.info(f"Executing step {step_id}: {action}")
        depends_on = step.get("depends_on", []) or []
        stage_reminder = str(step.get("_stage_reminder", ""))
        dep_ctx = self._gather_dep_context(depends_on)
        suspicious_deps = suspicious_dependency_ids(self.execution_context, depends_on)
        triggered_by_untrusted_context = bool(step.get("triggered_by_untrusted_context", False)) or bool(suspicious_deps)
        start = time.monotonic()

        pre_decision = self.hook_manager.run_pre_tool_use(HookContext(
            event="pre_tool_use",
            payload={
                "step_id": step_id,
                "action": action,
                "params_preview": preview_payload(params),
                "triggered_by_untrusted_context": triggered_by_untrusted_context,
                "suspicious_dependency_step_ids": suspicious_deps,
            },
            metadata={"step": step},
        ))
        if not pre_decision.allowed:
            result = StepResult(step_id, action, False, error=f"Hook blocked tool call: {pre_decision.reason}")
            self.execution_context[step_id] = result
            self._record_tool_call(step_id, action, params, result, start, policy_decision=None, hook_decision=pre_decision)
            return result

        policy_decision = self.tool_policy.check(
            action=action,
            params=params,
            triggered_by_untrusted_context=triggered_by_untrusted_context,
        )
        if not policy_decision.allowed:
            result = StepResult(
                step_id,
                action,
                False,
                error=f"Policy blocked tool call: {policy_decision.reason}",
                data={
                    "triggered_by_untrusted_context": triggered_by_untrusted_context,
                    "suspicious_dependency_step_ids": suspicious_deps,
                },
            )
            self.execution_context[step_id] = result
            self._record_tool_call(step_id, action, params, result, start, policy_decision)
            return result

        try:
            result = await self.dispatcher.dispatch(
                action,
                ToolInvocationContext(action=action, params=params, step=step,
                                      dep_context=dep_ctx, stage_reminder=stage_reminder),
            )
        except ToolDispatchError as e:
            result = StepResult(step_id, action, False, error=str(e))
        except Exception as e:
            logger.error(f"Step {step_id} failed: {e}")
            result = StepResult(step_id, action, False, error=str(e))

        result.step_id = step_id
        result.action = action
        if triggered_by_untrusted_context:
            result.data["triggered_by_untrusted_context"] = True
            result.data["suspicious_dependency_step_ids"] = suspicious_deps
        self._annotate_untrusted_output(action, result)
        self.execution_context[step_id] = result
        post_decision = self.hook_manager.run_post_tool_use(HookContext(
            event="post_tool_use",
            payload={
                "step_id": step_id,
                "action": action,
                "success": result.success,
                "error": result.error,
                "output_preview": result.output[:500],
            },
            metadata={"step": step, "result": result},
        ))
        if not post_decision.allowed:
            result.success = False
            result.error = f"Post-tool hook blocked result: {post_decision.reason}"
            self.execution_context[step_id] = result
        self._record_tool_call(step_id, action, params, result, start, policy_decision, hook_decision=post_decision)
        return result

    def _record_tool_call(self, step_id: int, action: str, params: dict, result: StepResult,
                          start: float, policy_decision=None, hook_decision=None) -> None:
        risk_level = policy_decision.risk.value if policy_decision is not None else "unknown"
        if hook_decision is not None and not hook_decision.allowed:
            decision = "hook_blocked"
        elif policy_decision is not None:
            decision = policy_decision.status
        else:
            decision = "not_checked"
        self.tool_calls.append(ToolCallRecord(
            tool_name=action,
            step_id=step_id,
            input_preview=preview_payload(params),
            output_preview=result.output[:500],
            success=result.success,
            error=result.error,
            latency_ms=int((time.monotonic() - start) * 1000),
            risk_level=risk_level,
            policy_decision=decision,
        ))

    def _register_local_tools(self) -> None:
        handlers = {
            "search_knowledge_base": lambda ctx: self._search(ctx.params),
            "read_document_detail": lambda ctx: self._read(ctx.params),
            "summarize_content": lambda ctx: self._summarize(ctx.params, ctx.dep_context, ctx.stage_reminder),
            "compare_concepts": lambda ctx: self._compare(ctx.params, ctx.dep_context, ctx.stage_reminder),
            "generate_research_report": lambda ctx: self._report(ctx.params, ctx.stage_reminder),
            "web_search": lambda ctx: self._web_search(ctx.params),
            "fetch_webpage": lambda ctx: self._fetch_webpage(ctx.params),
            "ask_user_clarification": lambda ctx: self._ask_user_clarification(ctx.params),
        }
        for tool_name, handler in handlers.items():
            if not self.dispatcher.has_handler(tool_name):
                self.dispatcher.register(tool_name, handler)

    def _annotate_untrusted_output(self, action: str, result: StepResult) -> None:
        if not result.output or action not in {"search_knowledge_base", "web_search", "fetch_webpage"}:
            return
        scan = scan_text(result.output)
        result.data["injection_scan"] = {
            "risk_score": scan.risk_score,
            "is_suspicious": scan.is_suspicious,
            "findings": [f.name for f in scan.findings],
        }
        if scan.is_suspicious:
            logger.warning("Suspicious untrusted context from %s: %s", action, scan.summary())

    def _gather_dep_context(self, depends_on: list[int]) -> str:
        parts = []
        for dep_id in depends_on:
            dep = self.execution_context.get(dep_id)
            if dep and dep.success:
                parts.append(f"[步骤{dep_id}结果] {dep.output[:500]}")
        return "\n\n".join(parts)

    async def _search(self, params: dict) -> StepResult:
        results = await self.retriever.search(
            query=params.get("query", ""),
            top_k_dense=config.rag.top_k_dense,
            top_k_sparse=config.rag.top_k_sparse,
            metadata_filter=params.get("filters"),
        )
        if not results:
            return StepResult(0, "search_knowledge_base", False, error="No results found")
        reranked = await self.reranker.rerank(params.get("query", ""), results, top_k=params.get("top_k", 5))
        context, citations = self.citation_tracer.build_citation_context(
            reranked,
            start_index=len(self.all_citations) + 1,
        )
        self.all_citations.update(citations)
        return StepResult(0, "search_knowledge_base", True, output=context,
                          data={"num_results": len(reranked), "citations": citations, "query": params.get("query")})

    async def _read(self, params: dict) -> StepResult:
        doc_id = params.get("doc_id", "")
        chunks = [c for c in self.retriever.chunks if c.doc_id == doc_id]
        if params.get("section"):
            chunks = [c for c in chunks if c.metadata.get("section_title", "").lower() == params["section"].lower()]
        if not chunks:
            return StepResult(0, "read_document_detail", False, error=f"Doc not found: {doc_id}")
        chunks.sort(key=lambda c: c.chunk_index)
        return StepResult(0, "read_document_detail", True,
                          output="\n\n".join(c.content for c in chunks)[:3000])

    async def _summarize(self, params: dict, dep_ctx: str, stage_reminder: str = "") -> StepResult:
        content = params.get("content", dep_ctx)
        focus = params.get("focus", "")
        prompt = f"摘要(≤{params.get('max_length',300)}字)"
        if focus: prompt += f"，关注: {focus}"
        user_msg = f"{prompt}\n\n{content[:4000]}"
        if stage_reminder:
            user_msg = f"{stage_reminder}\n\n---\n\n{user_msg}"
        r = await llm_client.chat([
            {"role": "system", "content": "你是精准的研究摘要助手。"},
            {"role": "user", "content": user_msg},
        ], temperature=0.2)
        return StepResult(0, "summarize_content", True, output=r["content"])

    async def _compare(self, params: dict, dep_ctx: str, stage_reminder: str = "") -> StepResult:
        prompt = f"对比: {', '.join(params.get('concepts', []))}"
        if params.get("dimensions"): prompt += f"\n维度: {', '.join(params['dimensions'])}"
        if dep_ctx: prompt += f"\n参考:\n{dep_ctx}"
        if stage_reminder:
            prompt = f"{stage_reminder}\n\n---\n\n{prompt}"
        r = await llm_client.chat([
            {"role": "system", "content": "你是技术分析专家，善于多维对比。"},
            {"role": "user", "content": prompt},
        ], temperature=0.3)
        return StepResult(0, "compare_concepts", True, output=r["content"])

    async def _report(self, params: dict, stage_reminder: str = "") -> StepResult:
        all_outputs = [r.output for _, r in sorted(self.execution_context.items()) if r.success and r.output]
        material = "\n\n---\n\n".join(all_outputs)
        fmt = {"brief": "简报(≤500字)", "detailed": "结构化报告(背景/发现/分析/结论)", "academic": "学术风格报告"}
        user_msg = f"主题: {params.get('topic','')}\n\n素材:\n{material[:8000]}\n\n引用:\n{self.citation_tracer.format_citations(self.all_citations)}\n\n请生成报告。"
        if stage_reminder:
            user_msg = f"{stage_reminder}\n\n---\n\n{user_msg}"
        r = await llm_client.chat([
            {"role": "system", "content": f"你是研究报告撰写专家。{fmt.get(params.get('format','detailed'), fmt['detailed'])}\n所有观点标注 [来源N]。"},
            {"role": "user", "content": user_msg},
        ], temperature=0.3, max_tokens=4096)
        return StepResult(0, "generate_research_report", True, output=r["content"], data={"citations": self.all_citations})

    async def _ask_user_clarification(self, params: dict) -> StepResult:
        return StepResult(0, "ask_user_clarification", True, output=params.get("question", ""),
                          data={"needs_user_input": True, "options": params.get("options", [])})

    async def _web_search(self, params: dict) -> StepResult:
        """
        执行 Web 搜索

        搜索结果格式化为带编号的列表，方便后续步骤引用。
        同时将搜索结果注册到 citation 体系中，保持引用链完整。
        """
        from rag.web_search import web_searcher

        query = params.get("query", "")
        search_type = params.get("search_type", "text")
        max_results = params.get("max_results", 8)

        results = await web_searcher.search(query, max_results=max_results, search_type=search_type)
        if not results:
            return StepResult(0, "web_search", False, error=f"未找到相关搜索结果: {query}")

        # 格式化搜索结果 + 注册引用
        parts = []
        for i, r in enumerate(results, len(self.all_citations) + 1):
            cid = str(i)
            self.all_citations[cid] = {
                "doc_title": r.title,
                "doc_id": f"web_{cid}",
                "chunk_id": f"web_{cid}",
                "section": "",
                "source": r.url,
                "author": r.metadata.get("source_name", ""),
                "year": r.metadata.get("date", "")[:4] if r.metadata.get("date") else "",
            }
            source_tag = {"arxiv": "📄", "scholar": "🎓", "github": "💻", "blog": "📝"}.get(r.source, "🔗")
            parts.append(f"[来源{cid}] {source_tag} {r.title}\n{r.snippet}\nURL: {r.url}")

        output = "\n\n---\n\n".join(parts)
        return StepResult(0, "web_search", True, output=output,
                          data={"num_results": len(results), "query": query,
                                "urls": [r.url for r in results], "citations": self.all_citations})

    async def _fetch_webpage(self, params: dict) -> StepResult:
        """
        抓取网页全文

        面试点: 为什么抓取后要截断到 max_content_length？
        答: 一个网页可能有 50K+ 字符，全部塞进上下文会:
            1. 挤占其他信息的空间 (token 预算有限)
            2. LLM 对超长输入的注意力衰减
            3. 增加 API 成本和延迟
        截断到 8000 字符 (~3000 tokens) 是合理的 trade-off
        """
        from rag.web_search import web_fetcher

        url = params.get("url", "")
        if not url:
            return StepResult(0, "fetch_webpage", False, error="未提供 URL")

        page = await web_fetcher.fetch(url)
        if not page:
            return StepResult(0, "fetch_webpage", False, error=f"无法抓取网页: {url}")

        # 注册引用
        cid = str(len(self.all_citations) + 1)
        self.all_citations[cid] = {
            "doc_title": page.title, "doc_id": f"web_{cid}", "chunk_id": f"web_{cid}",
            "section": "", "source": url, "author": "", "year": "",
        }

        output = f"[来源{cid}] {page.title}\nURL: {url}\n\n{page.content}"
        return StepResult(0, "fetch_webpage", True, output=output,
                          data={"url": url, "title": page.title, "content_length": page.content_length})
