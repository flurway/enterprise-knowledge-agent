"""
任务规划器 — Plan-and-Execute
"""
import json
import logging
from models.deepseek import llm_client
from config import config, get_current_date, get_current_year

logger = logging.getLogger(__name__)


def _build_planner_prompt() -> str:
    year = get_current_year()
    prev_year = str(int(year) - 1)
    json_example = """{
    "research_goal": "研究目标",
    "complexity": "simple|medium|complex",
    "steps": [
        {
            "step_id": 1,
            "action": "工具名",
            "description": "做什么",
            "input_params": {},
            "depends_on": [],
            "expected_output": "预期结果",
            "verification": "验证方式"
        }
    ]
}"""
    return f"""你是研究任务规划器。给定研究问题，制定分步执行计划。
当前日期: {get_current_date()}

## 规划原则
1. 每步是一个"可验证的阶段性成果"
2. 控制在 3-6 步
3. 步骤间有依赖关系
4. 最后一步必须是综合/生成报告

## 可用工具
- search_knowledge_base: 从已上传的知识库检索文档 (优先使用)
- web_search: 在互联网搜索信息 (知识库不足时使用，或需要最新信息时)
- fetch_webpage: 抓取某个网页的完整内容 (搜索结果摘要不够时深入阅读)
- read_document_detail: 深入阅读知识库中的某个文档
- summarize_content: 摘要
- compare_concepts: 对比
- generate_research_report: 生成报告
- ask_user_clarification: 向用户提问

## 工具使用策略
1. 知识库优先: 先用 search_knowledge_base 在已有文档中搜索
2. 搜索补充: 如果知识库结果不足或用户问的是最新信息，用 web_search 补充
3. 深入阅读: 搜索结果的摘要不够详细时，用 fetch_webpage 抓取原文
4. 不要同时发起太多搜索: 每次最多 2 个搜索步骤

## 时间感知
- 当前年份是 {year} 年
- 用户提到"最新/最近/近期"时，搜索 query 中应包含 "{year}" 或 "{prev_year}" 等年份关键词
- 示例: 用户问"最新的LLM进展" → query 应为 "LLM 进展 {year}" 而非 "LLM 进展"

## 输出格式 (严格JSON)
{json_example}"""


class TaskPlanner:
    async def create_plan(self, query: str, sub_questions: list[str] = None, context: str = "") -> dict:
        content = f"研究问题: {query}"
        if sub_questions:
            content += "\n子问题:\n" + "\n".join(f"- {q}" for q in sub_questions)
        if context:
            content += f"\n参考上下文:\n{context}"
        content += "\n\n请制定计划，返回JSON。"

        plan = await llm_client.get_json_response([
            {"role": "system", "content": _build_planner_prompt()},
            {"role": "user", "content": content},
        ])
        if not plan or "steps" not in plan:
            plan = self._default_plan(query)
        if len(plan.get("steps", [])) > config.agent.max_planning_steps:
            plan["steps"] = plan["steps"][:config.agent.max_planning_steps]
        logger.info(f"Plan: {plan.get('complexity','?')} complexity, {len(plan.get('steps',[]))} steps")
        return plan

    async def replan(self, original_plan: dict, completed_steps: list[dict],
                     failed_step: dict, failure_reason: str) -> dict:
        plan = await llm_client.get_json_response([
            {"role": "system", "content": _build_planner_prompt() + "\n\n你需要修改计划，因为某步失败了。只输出尚未完成的新步骤。"},
            {"role": "user", "content": (
                f"原计划: {json.dumps(original_plan, ensure_ascii=False)[:1000]}\n"
                f"已完成: {json.dumps(completed_steps, ensure_ascii=False)}\n"
                f"失败步骤: {json.dumps(failed_step, ensure_ascii=False)}\n"
                f"失败原因: {failure_reason}\n请返回新的JSON计划。"
            )},
        ])
        if not plan or "steps" not in plan:
            return original_plan
        logger.info(f"Replanned: {len(plan['steps'])} new steps")
        return plan

    def _default_plan(self, query: str) -> dict:
        return {
            "research_goal": query, "complexity": "simple",
            "steps": [
                {"step_id": 1, "action": "search_knowledge_base",
                 "description": f"检索'{query}'相关文档", "input_params": {"query": query, "top_k": 5},
                 "depends_on": [], "expected_output": "相关片段", "verification": "是否找到相关文档"},
                {"step_id": 2, "action": "generate_research_report",
                 "description": "生成回答", "input_params": {"topic": query, "format": "brief"},
                 "depends_on": [1], "expected_output": "结构化回答", "verification": "是否基于文档"},
            ],
        }
