"""
LangGraph-native orchestration for ResearchAgent.

The graph owns high-level control flow. Existing project components still own
domain behavior: tool policy, dispatcher, hooks, memory, answer validation, and
trace persistence.
"""
from __future__ import annotations

import logging
from typing import Any

from agent.executor import TaskExecutor
from agent.intent import (
    INTENT_CHITCHAT,
    INTENT_DIRECT_SEARCH,
    INTENT_FOLLOW_UP,
    INTENT_NEED_CLARIFY,
)
from agent.langgraph_adapter import LangGraphAdapter, ReliableAgentGraphState
from agent.runtime import AgentRuntime, PermissionMode, ReActLoop, RuntimeMode, RuntimeNode
from agent.stage_reminder import StageReminderManager, default_stage_constraints
from agent.state import AgentRun

logger = logging.getLogger(__name__)


class ResearchGraphState(ReliableAgentGraphState, total=False):
    original_query: str
    search_query: str
    sub_questions: list[str]
    memory: Any
    run: AgentRun
    runtime: AgentRuntime
    lt_ctx: str
    conflict_notices: list[dict]
    executor: TaskExecutor
    all_step_results: list[Any]
    failed_result: Any
    failed_step: dict[str, Any]
    reflection: dict[str, Any]
    hallucination_check: dict[str, Any]
    replan_count: int
    short_circuit: bool
    skip_synthesis: bool


class LangGraphResearchRuntime:
    def __init__(self, agent: Any, adapter: LangGraphAdapter | None = None):
        self.agent = agent
        self.adapter = adapter or LangGraphAdapter()

    async def chat(self, user_message: str, session_id: str = "default") -> dict:
        memory = self.agent.get_or_create_session(session_id)
        run = AgentRun(session_id=session_id, user_message=user_message)
        runtime = AgentRuntime(
            run,
            stage_reminder=StageReminderManager(constraints=default_stage_constraints()),
        )
        runtime.set_permission_mode(PermissionMode.AUTO, "LangGraph runtime can execute allowed low/medium-risk tools")

        initial_state: ResearchGraphState = {
            "session_id": session_id,
            "user_message": user_message,
            "original_query": user_message,
            "memory": memory,
            "run": run,
            "runtime": runtime,
            "all_step_results": [],
            "conflict_notices": [],
            "replan_count": 0,
        }
        graph = self.adapter.build(handlers=self._handlers())
        final_state = await graph.ainvoke(initial_state)
        return final_state["final_result"]

    def _handlers(self) -> dict[str, Any]:
        return {
            RuntimeNode.CLASSIFY.value: self._classify,
            "route_classified": self._route_passthrough,
            RuntimeNode.RETRIEVE_MEMORY.value: self._retrieve_memory,
            "route_intent": self._route_passthrough,
            RuntimeNode.DIRECT_SEARCH.value: self._direct_search,
            RuntimeNode.PLAN.value: self._plan,
            RuntimeNode.VALIDATE_PLAN.value: self._validate_plan,
            RuntimeNode.EXECUTE_STEP.value: self._execute_step,
            RuntimeNode.REFLECT.value: self._reflect,
            RuntimeNode.REPLAN.value: self._replan,
            RuntimeNode.SYNTHESIZE.value: self._synthesize,
            RuntimeNode.VALIDATE_ANSWER.value: self._validate_answer,
            RuntimeNode.SAVE_MEMORY.value: self._save_memory,
            RuntimeNode.FINISH.value: self._finish,
        }

    async def _classify(self, state: ResearchGraphState) -> dict[str, Any]:
        memory = state["memory"]
        run = state["run"]
        runtime = state["runtime"]
        user_message = state["user_message"]

        resolution = self.agent._try_resolve_memory_conflict_command(user_message, run)
        if resolution is not None:
            return {"result": resolution, "short_circuit": True}

        await self.agent._save_explicit_user_memory_candidates(user_message, state["session_id"], run)

        runtime.enter_node(RuntimeNode.CLASSIFY, "Classifying user intent")
        intent_result = await self.agent.intent_classifier.classify(user_message, memory)
        intent = intent_result.get("intent", INTENT_DIRECT_SEARCH)
        run.intent = intent

        if intent == INTENT_CHITCHAT:
            resp = await self.agent._handle_chitchat(user_message, memory)
            memory.add_turn("user", user_message, {"intent": intent})
            memory.add_turn("assistant", resp)
            return {
                "intent": intent,
                "result": {"response": resp, "intent": intent, "citations": {}},
                "short_circuit": True,
            }

        if intent == INTENT_NEED_CLARIFY:
            question = intent_result.get("clarify_question", "能否更具体地描述你的研究问题？")
            memory.add_turn("user", user_message, {"intent": intent})
            memory.add_turn("assistant", question)
            return {
                "intent": intent,
                "result": {
                    "response": question,
                    "intent": intent,
                    "citations": {},
                    "needs_clarification": True,
                    "clarification_question": question,
                    "missing_info": intent_result.get("missing_info", []),
                },
                "short_circuit": True,
            }

        if intent == INTENT_FOLLOW_UP:
            resolved = await self.agent.intent_classifier.resolve_follow_up(user_message, memory)
            re_intent = await self.agent.intent_classifier.classify(resolved)
            intent = re_intent.get("intent", INTENT_DIRECT_SEARCH)
            search_query = resolved
        else:
            search_query = intent_result.get("refined_query", user_message)

        run.intent = intent
        run.refined_query = search_query
        run.add_event("intent_classified", f"Intent classified as {intent}", refined_query=search_query)
        return {
            "intent": intent,
            "intent_result": intent_result,
            "search_query": search_query,
            "sub_questions": intent_result.get("sub_questions", []),
            "short_circuit": False,
        }

    async def _route_passthrough(self, state: ResearchGraphState) -> dict[str, Any]:
        return {}

    async def _retrieve_memory(self, state: ResearchGraphState) -> dict[str, Any]:
        runtime = state["runtime"]
        run = state["run"]
        runtime.enter_node(RuntimeNode.RETRIEVE_MEMORY, "Retrieving long-term memory")
        bundle = await self.agent._retrieve_long_term_memory(state["search_query"], run)
        run.add_event("long_term_memory_retrieved", "Long-term memory retrieval completed",
                      has_context=bool(bundle["context"]))
        return {
            "lt_ctx": bundle["context"],
            "conflict_notices": bundle["conflict_notices"],
        }

    async def _direct_search(self, state: ResearchGraphState) -> dict[str, Any]:
        runtime = state["runtime"]
        runtime.set_mode(RuntimeMode.EXECUTE, "Direct search can execute read-only retrieval tools")
        result = await self.agent._handle_direct_search(
            state["original_query"],
            state["search_query"],
            state["memory"],
            state.get("lt_ctx", ""),
            runtime,
        )
        return {"result": result}

    async def _plan(self, state: ResearchGraphState) -> dict[str, Any]:
        runtime = state["runtime"]
        runtime.set_mode(RuntimeMode.PLAN, "Creating plan before executing deep research")
        runtime.enter_node(RuntimeNode.PLAN, "Creating research plan", query=state["search_query"])
        plan = await self.agent.planner.create_plan(
            query=state["search_query"],
            sub_questions=state.get("sub_questions", []),
            context=state.get("lt_ctx", ""),
        )
        self.agent.reflector.reset()
        return {
            "plan": plan,
            "steps": list(plan.get("steps", [])),
            "executor": TaskExecutor(self.agent.retriever, hook_manager=self.agent.hook_manager),
            "all_step_results": [],
            "failed_result": None,
            "failed_step": {},
            "needs_replan": False,
        }

    async def _validate_plan(self, state: ResearchGraphState) -> dict[str, Any]:
        runtime = state["runtime"]
        plan = state["plan"]
        runtime.set_plan(plan)
        runtime.load_plan_as_todos(plan)
        runtime.set_mode(RuntimeMode.EXECUTE, "Executing todos derived from the research plan")
        runtime.enter_node(RuntimeNode.VALIDATE_PLAN, "Validating research plan")
        errors = runtime.validate_plan(plan)
        return {"plan_validation_errors": errors}

    async def _execute_step(self, state: ResearchGraphState) -> dict[str, Any]:
        runtime = state["runtime"]
        executor = state["executor"]
        steps = list(state.get("steps", []))
        steps_by_id = {int(step.get("step_id", 0) or 0): step for step in steps}
        loop = ReActLoop(runtime, max_iterations=max(len(steps) + 2, 3))
        loop_results = await loop.run_plan_todos(steps_by_id, executor.execute_step)

        all_results = list(state.get("all_step_results", []))
        all_results.extend(loop_results)
        clarification = next((r for r in loop_results if r.action == "ask_user_clarification"), None)
        if clarification:
            return {
                "all_step_results": all_results,
                "result": {
                    "response": clarification.output,
                    "needs_clarification": True,
                    "clarification_question": clarification.output,
                    "citations": {},
                    "plan": state["plan"],
                },
                "skip_synthesis": True,
                "needs_replan": False,
            }

        failed = next((r for r in loop_results if not r.success), None)
        failed_step = {}
        if failed:
            failed_step = steps_by_id.get(failed.step_id, {
                "step_id": failed.step_id,
                "action": failed.action,
                "input_params": {},
                "depends_on": [],
            })
        return {
            "all_step_results": all_results,
            "failed_result": failed,
            "failed_step": failed_step,
            "skip_synthesis": False,
        }

    async def _reflect(self, state: ResearchGraphState) -> dict[str, Any]:
        runtime = state["runtime"]
        failed = state.get("failed_result")
        if state.get("skip_synthesis"):
            return {"needs_replan": False}

        if failed:
            replan_count = int(state.get("replan_count", 0))
            max_replans = self.agent.reflector.max_rounds
            if replan_count < max_replans:
                runtime.enter_node(RuntimeNode.REPLAN, f"Replanning after failed step {failed.step_id}",
                                   failed_step=failed.step_id, error=failed.error)
                return {"needs_replan": True}
            logger.warning("Step failed and max replans reached, continuing to synthesis")
            return {"needs_replan": False}

        runtime.enter_node(RuntimeNode.REFLECT, "Reflecting on gathered step results")
        reflection = await self.agent.reflector.reflect(
            state["search_query"],
            state.get("all_step_results", []),
            state["plan"],
        )
        all_results = list(state.get("all_step_results", []))
        if not reflection.get("is_sufficient", True):
            executor = state["executor"]
            for action in reflection.get("suggested_actions", [])[:2]:
                supplementary = await executor.execute_step({
                    "step_id": len(all_results) + 1,
                    "action": action.get("action", "search_knowledge_base"),
                    "input_params": action.get("params", {}),
                    "depends_on": [],
                })
                all_results.append(supplementary)
        return {
            "reflection": reflection,
            "all_step_results": all_results,
            "needs_replan": False,
        }

    async def _replan(self, state: ResearchGraphState) -> dict[str, Any]:
        failed = state["failed_result"]
        new_plan = await self.agent.planner.replan(
            original_plan=state["plan"],
            completed_steps=[r.to_dict() for r in state.get("all_step_results", []) if r.success],
            failed_step=state.get("failed_step", {}),
            failure_reason=failed.error,
        )
        steps = list(new_plan.get("steps", []))
        replan_count = int(state.get("replan_count", 0)) + 1
        state["run"].add_event("replan_applied", f"Replan #{replan_count} applied", failed_step=failed.step_id)
        return {
            "plan": new_plan,
            "steps": steps,
            "replan_count": replan_count,
            "failed_result": None,
            "failed_step": {},
            "needs_replan": False,
        }

    async def _synthesize(self, state: ResearchGraphState) -> dict[str, Any]:
        runtime = state["runtime"]
        executor = state["executor"]
        all_results = list(state.get("all_step_results", []))
        runtime.enter_node(RuntimeNode.SYNTHESIZE, "Generating final research report")
        report = await executor.execute_step({
            "step_id": 99,
            "action": "generate_research_report",
            "input_params": {
                "topic": state["search_query"],
                "findings": [r.output[:200] for r in all_results if r.success],
                "format": "detailed",
            },
            "depends_on": [r.step_id for r in all_results],
        })
        source_docs = "\n".join(r.output for r in all_results if r.success)
        hallucination_check = await self.agent.reflector.check_hallucination(report.output, source_docs[:6000])

        response = report.output
        if hallucination_check.get("hallucination_rate", 0) > 0.3:
            response += "\n\n⚠️ 部分内容可能缺乏充分文档支撑，建议进一步验证。"
        runtime.run.tool_calls.extend(executor.tool_calls)
        runtime.run.add_event("deep_research_completed",
                              f"Deep research completed with {len(executor.all_citations)} citation(s)")
        return {
            "result": {
                "response": response,
                "citations": executor.all_citations,
                "plan": state["plan"],
                "reflection": state.get("reflection", {}),
                "hallucination_check": hallucination_check,
            },
            "hallucination_check": hallucination_check,
        }

    async def _validate_answer(self, state: ResearchGraphState) -> dict[str, Any]:
        result = dict(state.get("result", {}))
        self.agent._attach_conflict_aware_answer(result, state.get("conflict_notices", []), state["run"])
        return {"result": result}

    async def _save_memory(self, state: ResearchGraphState) -> dict[str, Any]:
        result = state.get("result", {})
        memory = state["memory"]
        memory.add_turn("user", state["original_query"], {"intent": state.get("intent", "")})
        memory.add_turn("assistant", result.get("response", ""))
        if len(memory.turns) > memory.summary_threshold:
            from models.deepseek import llm_client
            await memory.compress_history(llm_client)

        decision = self.agent.memory_write_gate.decide(state.get("search_query", state["original_query"]), result, state["session_id"])
        state["run"].add_event("memory_write_decision", decision.reason, **decision.to_dict())
        if decision.should_write:
            state["runtime"].enter_node(RuntimeNode.SAVE_MEMORY, "Saving research summary to long-term memory")
            await self.agent._save_to_long_term_memory(
                state.get("search_query", state["original_query"]),
                result.get("response", ""),
                state["session_id"],
                decision,
            )
        result["intent"] = state.get("intent", "")
        return {"result": result}

    async def _finish(self, state: ResearchGraphState) -> dict[str, Any]:
        result = self.agent._finish_result(state.get("result", {}), state["run"])
        return {"final_result": result}

