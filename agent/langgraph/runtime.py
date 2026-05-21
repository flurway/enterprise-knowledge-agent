from __future__ import annotations

from typing import Any

from agent.langgraph.graph import ResearchGraphBuilder
from agent.langgraph.nodes import ResearchGraphNodes
from agent.langgraph.state import ResearchGraphState
from agent.runtime import AgentRuntime, PermissionMode
from agent.stage_reminder import StageReminderManager, default_stage_constraints
from agent.state import AgentRun


class LangGraphResearchRuntime:
    def __init__(self, agent: Any, graph_builder: ResearchGraphBuilder | None = None):
        self.agent = agent
        self.graph_builder = graph_builder or ResearchGraphBuilder()
        self.nodes = ResearchGraphNodes(agent)

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
        graph = self.graph_builder.build(handlers=self.nodes.handlers())
        final_state = await graph.ainvoke(initial_state)
        return final_state["final_result"]

