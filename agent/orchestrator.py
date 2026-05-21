"""
ResearchGPT 主编排器

修复记录:
- [BUG FIX] _handle_deep_research 中 replan 失效问题
  原因: for step in plan["steps"] 的迭代器在循环开始时绑定了原始列表，
        plan 被重新赋值后迭代器仍走老列表，导致新计划永远不会被执行。
  修复: 改为 while + index 循环，replan 后重置 steps 列表和 index，
        从新计划的第一步重新开始执行。
"""
import logging
import numpy as np
from typing import Optional
from config import config, get_current_date, get_current_year
from models.deepseek import llm_client
from memory.short_term import ConversationMemory
from memory.long_term import LongTermMemory
from memory.write_gate import MemoryWriteGate
from memory.extractor import ExplicitMemoryExtractor
from memory.retrieval_budget import MemoryRetrievalBudget
from rag.retriever import HybridRetriever
from agent.intent import (IntentClassifier, INTENT_CHITCHAT, INTENT_NEED_CLARIFY,
                           INTENT_FOLLOW_UP, INTENT_DIRECT_SEARCH, INTENT_DEEP_RESEARCH)
from agent.planner import TaskPlanner
from agent.executor import TaskExecutor
from agent.reflector import Reflector
from agent.state import AgentRun
from agent.trace_store import AgentTraceStore
from agent.answer_validator import validate_answer
from agent.runtime import AgentRuntime, ReActLoop, RuntimeNode
from agent.runtime import RuntimeMode, PermissionMode
from agent.stage_reminder import StageReminderManager, StageConstraints, default_stage_constraints
from agent.hooks import HookContext, HookManager

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    return f"""你是 ResearchGPT，专业的 AI 研究助手。
当前日期: {get_current_date()}

## 回答原则
- 基于检索到的文档回答，不编造信息
- 关键观点标注引用 [来源N]
- 简单问题直接精准回答；复杂问题结构化分析
- 如果检索不到信息，诚实说明
- 注意信息时效性：优先引用近 1-2 年的资料，标注资料年份
- 当前是 {get_current_year()} 年，用户问"最新/最近"指的是 {get_current_year()} 年前后"""


class ResearchAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.intent_classifier = IntentClassifier()
        self.planner = TaskPlanner()
        self.reflector = Reflector(
            confidence_threshold=config.agent.confidence_threshold,
            max_rounds=config.agent.max_reflection_rounds,
        )
        self.long_term_memory = LongTermMemory(
            index_path=config.memory.long_term_index_path,
            embedding_dim=config.rag.embedding_dim,
        )
        self.memory_write_gate = MemoryWriteGate()
        self.memory_extractor = ExplicitMemoryExtractor()
        self.memory_retrieval_budget = MemoryRetrievalBudget.from_config(config.memory)
        self.trace_store = (
            AgentTraceStore(config.agent.trace_db_path)
            if config.agent.enable_trace_persistence
            else None
        )
        self.hook_manager = HookManager()
        self.sessions: dict[str, ConversationMemory] = {}

    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(
                max_turns=config.memory.max_short_term_turns,
                summary_threshold=config.memory.summary_threshold,
            )
            self.sessions[session_id].session_id = session_id
        return self.sessions[session_id]

    async def chat(self, user_message: str, session_id: str = "default") -> dict:
        if config.agent.runtime_backend == "langgraph":
            from agent.langgraph.runtime import LangGraphResearchRuntime
            return await LangGraphResearchRuntime(self).chat(user_message, session_id)

        memory = self.get_or_create_session(session_id)
        run = AgentRun(session_id=session_id, user_message=user_message)
        stage_reminder = StageReminderManager(constraints=default_stage_constraints())
        runtime = AgentRuntime(run, stage_reminder=stage_reminder)
        runtime.set_permission_mode(PermissionMode.AUTO, "Research runtime can execute allowed low/medium-risk tools")

        conflict_resolution = self._try_resolve_memory_conflict_command(user_message, run)
        if conflict_resolution is not None:
            return self._finish_result(conflict_resolution, run)

        await self._save_explicit_user_memory_candidates(user_message, session_id, run)

        # ===== 意图判断 =====
        runtime.enter_node(RuntimeNode.CLASSIFY, "Classifying user intent")
        intent_result = await self.intent_classifier.classify(user_message, memory)
        intent = intent_result.get("intent", INTENT_DIRECT_SEARCH)
        run.intent = intent
        logger.info(f"[{session_id}] Intent: {intent} | Query: {user_message[:50]}")

        # 闲聊
        if intent == INTENT_CHITCHAT:
            resp = await self._handle_chitchat(user_message, memory)
            memory.add_turn("user", user_message, {"intent": intent})
            memory.add_turn("assistant", resp)
            return self._finish_result({"response": resp, "intent": intent, "citations": {}}, run)

        # 需要澄清
        if intent == INTENT_NEED_CLARIFY:
            q = intent_result.get("clarify_question", "能否更具体地描述你的研究问题？")
            memory.add_turn("user", user_message, {"intent": intent})
            memory.add_turn("assistant", q)
            return self._finish_result({
                "response": q, "intent": intent, "citations": {},
                "needs_clarification": True, "clarification_question": q,
                "missing_info": intent_result.get("missing_info", []),
            }, run)

        # 追问 → 指代消解
        if intent == INTENT_FOLLOW_UP:
            resolved = await self.intent_classifier.resolve_follow_up(user_message, memory)
            re_intent = await self.intent_classifier.classify(resolved)
            intent = re_intent.get("intent", INTENT_DIRECT_SEARCH)
            search_query = resolved
        else:
            search_query = intent_result.get("refined_query", user_message)
        run.refined_query = search_query
        run.add_event("intent_classified", f"Intent classified as {intent}", refined_query=search_query)

        # ===== 长期记忆 =====
        runtime.enter_node(RuntimeNode.RETRIEVE_MEMORY, "Retrieving long-term memory")
        lt_bundle = await self._retrieve_long_term_memory(search_query, run)
        lt_ctx = lt_bundle["context"]
        conflict_notices = lt_bundle["conflict_notices"]
        run.add_event("long_term_memory_retrieved", "Long-term memory retrieval completed", has_context=bool(lt_ctx))

        # ===== 执行研究 =====
        if intent == INTENT_DIRECT_SEARCH:
            runtime.set_mode(RuntimeMode.EXECUTE, "Direct search can execute read-only retrieval tools")
            result = await self._handle_direct_search(user_message, search_query, memory, lt_ctx, runtime)
        else:
            result = await self._handle_deep_research(
                user_message, search_query,
                intent_result.get("sub_questions", []), memory, lt_ctx, runtime,
            )
        self._attach_conflict_aware_answer(result, conflict_notices, run)

        # ===== 更新记忆 =====
        memory.add_turn("user", user_message, {"intent": intent})
        memory.add_turn("assistant", result["response"])
        if len(memory.turns) > memory.summary_threshold:
            await memory.compress_history(llm_client)
        memory_decision = self.memory_write_gate.decide(search_query, result, session_id)
        run.add_event("memory_write_decision", memory_decision.reason, **memory_decision.to_dict())
        if memory_decision.should_write:
            runtime.enter_node(RuntimeNode.SAVE_MEMORY, "Saving research summary to long-term memory")
            await self._save_to_long_term_memory(search_query, result["response"], session_id, memory_decision)

        result["intent"] = intent
        return self._finish_result(result, run)

    def _finish_result(self, result: dict, run: AgentRun, success: bool = True) -> dict:
        if result.get("citations"):
            run.add_event("node_entered", "Entered runtime node: validate_answer", node=RuntimeNode.VALIDATE_ANSWER.value)
            validation = validate_answer(result.get("response", ""), result.get("citations", {}))
            result["answer_validation"] = validation.to_dict()
            if not validation.ok:
                run.add_event("answer_validation_failed", "Answer-level validation found issues", **validation.to_dict())
        answer_decision = self.hook_manager.run_answer_ready(HookContext(
            event="answer_ready",
            payload={
                "run_id": run.run_id,
                "intent": run.intent,
                "has_citations": bool(result.get("citations")),
                "response_preview": result.get("response", "")[:500],
                "answer_validation": result.get("answer_validation"),
            },
            metadata={"result": result, "run": run},
        ))
        if not answer_decision.allowed:
            run.add_event("answer_ready_hook_blocked", "Answer-ready hook blocked final answer",
                          reason=answer_decision.reason, hook_metadata=answer_decision.metadata)
            result["answer_hook_blocked"] = {
                "reason": answer_decision.reason,
                "metadata": answer_decision.metadata,
            }
        run.add_event("node_entered", "Entered runtime node: finish", node=RuntimeNode.FINISH.value)
        run.finish(success)
        trace = run.to_dict()
        result["run_id"] = run.run_id
        result["trace"] = trace
        self._persist_run(run)
        return result

    def _persist_run(self, run: AgentRun) -> None:
        if self.trace_store is None:
            return
        try:
            self.trace_store.save_run(run)
        except Exception as e:
            logger.warning(f"Failed to persist agent trace {run.run_id}: {e}")

    async def _handle_chitchat(self, message: str, memory: ConversationMemory) -> str:
        msgs = memory.build_context_messages(
            system_prompt="你是 ResearchGPT，友好的AI研究助手。简短回应闲聊，温和引导回研究话题。",
            current_query=message,
        )
        r = await llm_client.chat(msgs, temperature=0.7)
        return r["content"]

    async def _handle_direct_search(self, original_query: str, search_query: str,
                                     memory: ConversationMemory, lt_ctx: str,
                                     runtime: AgentRuntime | None = None) -> dict:
        """
        直接检索 — 智能信息源选择

        策略 (不只是 fallback，而是动态互补):
        1. 知识库有索引 → 先检索知识库
        2. 知识库结果充分 (≥3条) → 直接用知识库
        3. 知识库结果不足 (<3条) 或无索引 → Web 搜索补充
        4. 两者结果合并，知识库在前 (用户精选) + 搜索在后 (广度补充)
        
        面试点: 为什么不每次都搜索？
        答: 1. 搜索有延迟 (500ms~2s)，知识库检索 <50ms
            2. 知识库文档是用户精选的高质量内容，搜索结果质量参差不齐
            3. 只在知识库不够时搜索，平衡了速度和覆盖度
        """
        if runtime:
            runtime.enter_node(RuntimeNode.DIRECT_SEARCH, "Running direct search flow", query=search_query)
        executor = TaskExecutor(self.retriever, hook_manager=self.hook_manager)
        rag_ctx = ""
        kb_result_count = 0

        # Step 1: 知识库检索
        if self.retriever.faiss_index is not None and self.retriever.faiss_index.ntotal > 0:
            sr = await executor.execute_step({
                "step_id": 1, "action": "search_knowledge_base",
                "input_params": {"query": search_query, "top_k": config.rag.top_k_rerank},
                "depends_on": [],
            })
            if sr.success:
                rag_ctx = sr.output
                kb_result_count = sr.data.get("num_results", 0)

        # Step 2: 判断是否需要 Web 搜索补充
        #   - 知识库为空或无结果 → 必须搜索
        #   - 知识库结果少 (<3条) → 搜索补充
        #   - 知识库结果充分 (≥3条) → 不搜索
        need_web_search = kb_result_count < 3

        if need_web_search:
            reason = "知识库无索引" if kb_result_count == 0 and (self.retriever.faiss_index is None or self.retriever.faiss_index.ntotal == 0) else \
                     "知识库无命中" if kb_result_count == 0 else \
                     f"知识库结果不足({kb_result_count}条)"
            logger.info(f"Triggering web search: {reason}")

            web_sr = await executor.execute_step({
                "step_id": 2, "action": "web_search",
                "input_params": {"query": search_query, "max_results": 8},
                "depends_on": [],
            })
            if web_sr.success:
                if rag_ctx:
                    # 合并: 知识库在前 + 搜索在后
                    rag_ctx = f"{rag_ctx}\n\n===== Web 搜索补充 =====\n\n{web_sr.output}"
                else:
                    rag_ctx = web_sr.output

        # Step 3: 检索失败且无长期记忆时，生成基于模型知识的回答
        if not rag_ctx and not lt_ctx:
            msgs = memory.build_context_messages(
                system_prompt=_build_system_prompt(), current_query=original_query, rag_context=lt_ctx,
            )
            r = await llm_client.chat(msgs)
            if runtime:
                runtime.run.tool_calls.extend(executor.tool_calls)
                runtime.run.add_event("fallback_answer", "Answered from model knowledge because retrieval returned no context")
            return {"response": r["content"] + "\n\n(注: 知识库和搜索均未找到相关信息，以上基于模型自身知识)", "citations": {}}

        # 有长期记忆，根据历史研究参考给出回答
        if lt_ctx:
            rag_ctx = f"[历史研究参考]\n{lt_ctx}\n\n{rag_ctx}"
        msgs = memory.build_context_messages(
            system_prompt=_build_system_prompt(), current_query=original_query, rag_context=rag_ctx,
        )
        r = await llm_client.chat(msgs)
        if runtime:
            runtime.run.tool_calls.extend(executor.tool_calls)
            runtime.run.add_event("direct_search_completed", f"Direct search completed with {len(executor.all_citations)} citation(s)")
        return {"response": r["content"], "citations": executor.all_citations}

    async def _handle_deep_research(self, original_query: str, search_query: str,
                                     sub_questions: list[str], memory: ConversationMemory,
                                     lt_ctx: str, runtime: AgentRuntime | None = None) -> dict:
        """
        深度研究 — Plan-and-Execute

        [BUG FIX] 旧代码用 for step in plan["steps"] 循环，replan 后新计划
        不会被执行。改为 while + index，replan 时重置 steps 和 index。
        """
        # --- 规划 ---
        if runtime:
            runtime.set_mode(RuntimeMode.PLAN, "Creating plan before executing deep research")
            runtime.enter_node(RuntimeNode.PLAN, "Creating research plan", query=search_query)
        plan = await self.planner.create_plan(
            query=search_query, sub_questions=sub_questions, context=lt_ctx,
        )
        if runtime:
            runtime.set_plan(plan)
            runtime.load_plan_as_todos(plan)
            runtime.set_mode(RuntimeMode.EXECUTE, "Executing todos derived from the research plan")
            runtime.enter_node(RuntimeNode.VALIDATE_PLAN, "Validating research plan")
            runtime.validate_plan(plan)

        # --- 执行 (修复后的循环) ---
        executor = TaskExecutor(self.retriever, hook_manager=self.hook_manager)
        self.reflector.reset()
        all_step_results: list = []
        replan_count = 0
        max_replans = config.agent.max_reflection_rounds

        steps = list(plan.get("steps", []))
        while True:
            steps_by_id = {int(step.get("step_id", 0) or 0): step for step in steps}
            if runtime:
                loop = ReActLoop(runtime, max_iterations=max(len(steps) + 2, 3))
                loop_results = await loop.run_plan_todos(steps_by_id, executor.execute_step)
            else:
                loop_results = []
                for step in steps:
                    loop_results.append(await executor.execute_step(step))

            all_step_results.extend(loop_results)

            clarification = next((r for r in loop_results if r.action == "ask_user_clarification"), None)
            if clarification:
                if runtime:
                    runtime.run.tool_calls.extend(executor.tool_calls)
                return {
                    "response": clarification.output,
                    "needs_clarification": True,
                    "clarification_question": clarification.output,
                    "citations": {}, "plan": plan,
                }

            failed_result = next((r for r in loop_results if not r.success), None)
            if not failed_result:
                break

            failed_step = steps_by_id.get(failed_result.step_id, {
                "step_id": failed_result.step_id,
                "action": failed_result.action,
                "input_params": {},
                "depends_on": [],
            })
            if replan_count >= max_replans:
                logger.warning(f"Step failed and max replans ({max_replans}) reached, skipping")
                break

            logger.warning(f"Step {failed_result.step_id} failed: {failed_result.error}, triggering replan #{replan_count+1}")
            if runtime:
                runtime.enter_node(RuntimeNode.REPLAN, f"Replanning after failed step {failed_result.step_id}",
                                   failed_step=failed_result.step_id, error=failed_result.error)
            new_plan = await self.planner.replan(
                original_plan=plan,
                completed_steps=[r.to_dict() for r in all_step_results if r.success],
                failed_step=failed_step,
                failure_reason=failed_result.error,
            )
            new_steps = new_plan.get("steps", [])
            if not new_steps:
                break

            plan = new_plan
            steps = list(new_steps)
            replan_count += 1
            if runtime:
                runtime.run.add_event("replan_applied", f"Replan #{replan_count} applied", failed_step=failed_result.step_id)
                runtime.set_plan(plan)
                runtime.load_plan_as_todos(plan)
                runtime.enter_node(RuntimeNode.VALIDATE_PLAN, "Validating replanned steps")
                runtime.validate_plan(plan)
            logger.info(f"Replan applied: {len(steps)} new steps, restarting execution")

        # --- 反思 ---
        if runtime:
            runtime.enter_node(RuntimeNode.REFLECT, "Reflecting on gathered step results")
        reflection = await self.reflector.reflect(search_query, all_step_results, plan)

        if not reflection.get("is_sufficient", True):
            for action in reflection.get("suggested_actions", [])[:2]:
                supp = await executor.execute_step({
                    "step_id": len(all_step_results) + 1,
                    "action": action.get("action", "search_knowledge_base"),
                    "input_params": action.get("params", {}),
                    "depends_on": [],
                })
                all_step_results.append(supp)

        # --- 生成报告 ---
        if runtime:
            runtime.enter_node(RuntimeNode.SYNTHESIZE, "Generating final research report")
        report = await executor.execute_step({
            "step_id": 99, "action": "generate_research_report",
            "input_params": {
                "topic": search_query,
                "findings": [r.output[:200] for r in all_step_results if r.success],
                "format": "detailed",
            },
            "depends_on": [r.step_id for r in all_step_results],
        })

        # --- 幻觉检测 ---
        source_docs = "\n".join(r.output for r in all_step_results if r.success)
        hall_check = await self.reflector.check_hallucination(report.output, source_docs[:6000])

        response = report.output
        if hall_check.get("hallucination_rate", 0) > 0.3:
            response += "\n\n⚠️ 部分内容可能缺乏充分文档支撑，建议进一步验证。"
        if runtime:
            runtime.run.tool_calls.extend(executor.tool_calls)
            runtime.run.add_event("deep_research_completed", f"Deep research completed with {len(executor.all_citations)} citation(s)")

        return {
            "response": response,
            "citations": executor.all_citations,
            "plan": plan,
            "reflection": reflection,
            "hallucination_check": hall_check,
        }

    async def _retrieve_long_term_memory(self, query: str, run: AgentRun | None = None) -> dict:
        try:
            qe = self.retriever.get_embeddings([query])[0]
            memories = await self.long_term_memory.search(
                query_embedding=qe,
                top_k=max(config.memory.max_long_term_results * 4, config.memory.max_long_term_results),
            )
            selected = self.memory_retrieval_budget.select(memories)
            conflict_notices = self._find_relevant_memory_conflicts(memories)
            if run is not None:
                run.add_event(
                    "long_term_memory_budget_applied",
                    "Long-term memory retrieval budget applied",
                    candidates=len(memories),
                    selected=len(selected),
                    selected_types=[m.get("memory_type") for m in selected],
                    budget=self.memory_retrieval_budget.to_dict(),
                    filtered_conflicts=len(conflict_notices),
                )
            parts = [m["content"] for m in selected]
            return {
                "context": "\n\n".join(parts) if parts else "",
                "selected": selected,
                "conflict_notices": conflict_notices,
            }
        except Exception as e:
            logger.warning(f"Long-term memory retrieval failed: {e}")
            return {"context": "", "selected": [], "conflict_notices": []}

    def _find_relevant_memory_conflicts(self, memories: list[dict]) -> list[dict]:
        candidate_indices = {m.get("index") for m in memories if m.get("status") == "conflicted"}
        notices = []
        for review in self.long_term_memory.get_conflict_reviews(status="unresolved", limit=10):
            if review.get("existing_index") in candidate_indices or review.get("incoming_index") in candidate_indices:
                notices.append(review)
        return notices[:3]

    def _attach_conflict_aware_answer(self, result: dict, conflict_notices: list[dict], run: AgentRun) -> None:
        if not conflict_notices:
            return
        result["memory_conflicts"] = conflict_notices
        result["needs_memory_confirmation"] = True
        preview = conflict_notices[0]
        note = (
            "\n\n⚠️ 我发现相关长期记忆存在尚未解决的冲突。"
            f"冲突 #{preview['conflict_index']}：旧记忆和新记忆结论不一致。"
            "你可以回复“采用旧记忆”“采用新记忆”或“保留两者”来处理。"
        )
        result["response"] = f"{result.get('response', '')}{note}"
        run.add_event(
            "memory_conflict_notice_attached",
            "Attached unresolved memory conflict notice to answer",
            conflicts=[c.get("conflict_index") for c in conflict_notices],
        )

    def _try_resolve_memory_conflict_command(self, user_message: str, run: AgentRun) -> dict | None:
        resolution = self._parse_memory_conflict_resolution(user_message)
        if resolution is None:
            return None
        pending = self.long_term_memory.get_conflict_reviews(status="unresolved", limit=1)
        if not pending:
            return None
        conflict_index = pending[0]["conflict_index"]
        resolved = self.long_term_memory.resolve_conflict(conflict_index, resolution)
        run.add_event(
            "memory_conflict_resolution_requested",
            "User requested memory conflict resolution",
            conflict_index=conflict_index,
            resolution=resolution,
            resolved=resolved,
        )
        if resolved.get("resolved"):
            response = (
                f"已处理记忆冲突 #{conflict_index}：{self._resolution_label(resolution)}。"
                "后续长期记忆检索会按这个状态使用。"
            )
        else:
            response = f"我尝试处理记忆冲突 #{conflict_index}，但没有成功：{resolved.get('reason', 'unknown error')}"
        return {
            "response": response,
            "intent": "memory_conflict_review",
            "citations": {},
            "memory_conflict_resolution": resolved,
        }

    @staticmethod
    def _parse_memory_conflict_resolution(message: str) -> str | None:
        normalized = message.strip().lower()
        if any(token in normalized for token in ["采用新", "使用新", "以新的为准", "用新的", "use incoming", "use new"]):
            return "use_incoming"
        if any(token in normalized for token in ["采用旧", "使用旧", "以旧的为准", "用旧的", "use existing", "use old"]):
            return "use_existing"
        if any(token in normalized for token in ["保留两者", "都保留", "keep both"]):
            return "keep_both"
        return None

    @staticmethod
    def _resolution_label(resolution: str) -> str:
        labels = {
            "use_incoming": "采用新记忆",
            "use_existing": "采用旧记忆",
            "keep_both": "保留两者",
        }
        return labels.get(resolution, resolution)

    async def _save_to_long_term_memory(self, query: str, response: str, session_id: str, decision=None):
        try:
            emb = self.retriever.get_embeddings([query])[0]
            memory_type = decision.memory_type if decision else "research_summary"
            metadata = decision.metadata if decision else {}
            confidence = decision.confidence if decision else 1.0
            await self.long_term_memory.add_memory(
                content=f"研究问题: {query}\n结论摘要: {response[:500]}",
                memory_type=memory_type, embedding=emb,
                session_id=session_id, keywords=query.split()[:5],
                metadata=metadata, confidence=confidence,
            )
        except Exception as e:
            logger.warning(f"Failed to save long-term memory: {e}")

    async def _save_explicit_user_memory_candidates(self, user_message: str, session_id: str, run: AgentRun) -> None:
        candidates = self.memory_extractor.extract(user_message, session_id=session_id)
        run.add_event("memory_candidates_extracted", f"Extracted {len(candidates)} explicit memory candidate(s)",
                      count=len(candidates), candidates=[c.to_dict() for c in candidates])
        if not candidates:
            return
        for candidate in candidates:
            try:
                emb = self.retriever.get_embeddings([candidate.content])[0]
                await self.long_term_memory.add_memory(
                    content=candidate.content,
                    memory_type=candidate.memory_type,
                    embedding=emb,
                    session_id=session_id,
                    keywords=candidate.keywords,
                    metadata=candidate.metadata,
                    confidence=candidate.confidence,
                )
            except Exception as e:
                logger.warning(f"Failed to save explicit user memory candidate: {e}")
