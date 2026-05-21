# Agent Runtime Reliability Notes

中文锚点：Agent run trace 应该记录 tool call 字段，包括 tool name、step id、input preview、output preview、latency、success flag、error message、risk level 和 policy decision。

## Plan-and-Execute Runtime

Reliable research agents should not rely on a free-form prompt loop alone. A production-like runtime records an `AgentRun`, decomposes the user task into `PlanStep` objects, executes tools through a policy layer, and stores trace events for later inspection.

The recommended lifecycle is:

1. classify the user intent;
2. retrieve relevant memory;
3. create a bounded plan;
4. validate each step and its dependencies;
5. execute the selected tool;
6. verify step output;
7. replan only when the failure reason is actionable;
8. synthesize the final answer with citations;
9. persist trace and memory updates.

## Replan Failure Mode

A common Python bug appears when code uses `for step in plan["steps"]` and then replaces `plan` during replanning. The iterator still points to the old list. A safer implementation keeps an explicit `steps` list and `step_idx` index, then resets both when a new plan is accepted.

## Trace Requirements

Every tool call should record the tool name, step id, input preview, output preview, latency, success flag, error message, risk level, and policy decision. This makes agent failures inspectable instead of anecdotal.
