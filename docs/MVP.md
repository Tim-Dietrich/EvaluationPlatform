# Harbor Integration MVP

## Goal

Use Harbor to run one Self-Collaboration code-generation attempt against one
NL2RepoBench task and record the generated files, logs, and benchmark score.

Harbor is the orchestrator. Do not build a custom platform.

## Implement

1. Install and pin Harbor.
2. Create a thin Harbor agent adapter that:
   - invokes Self-Collaboration using its existing entry point;
   - supplies the task instruction and Harbor workspace;
   - passes credentials only through environment variables.
3. Reference NL2RepoBench through Harbor's own `nl2repobench/nl2repobench`
   registry dataset (pinned by content digest), filtered to the `math-verify`
   task. Harbor prepares the workspace and runs the evaluator itself; do not
   hand-build a parallel task definition.
4. Add one experiment configuration and document one command to run it.
5. Verify that Harbor retains the generated workspace, logs, evaluator output,
   and score.

Keep the Self-Collaboration submodule independent. Do not use NL2RepoBench's
OpenHands generation path, because generation is provided by Self-Collaboration.

## Done when

One documented command completes this flow:

````
text Harbor -> prepare one NL2RepoBench task -> run Self-Collaboration -> evaluate the generated workspace -> retain generated files, logs, and score
````


## Out of scope

Do not build a custom orchestrator, protocol, database, dashboard, artifact
system, scheduler, retry system, or generalized plugin framework. Do not modify
Harbor core or broadly refactor either submodule.

If an issue does not prevent the one-task run, document it and continue. Stop
once the flow above works.