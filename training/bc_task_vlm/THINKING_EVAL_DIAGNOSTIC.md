# Qwen Thinking Evaluation Diagnostic

## Technical summary

The out-of-box Qwen3-VL-8B-Thinking result is not a trustworthy capability baseline. The dominant failure was that the model consumed the 2,048-token completion budget with private reasoning and never reached a tool call. This is a generation-budget failure surfaced by the parser, not evidence that the configured Qwen tool parser is fundamentally incompatible.

The issue directly threatens future Thinking SFT evaluations if they retain the same 2,048-token limit. It does not directly corrupt teacher-forced SFT training, which does not use the vLLM response parser. However, a post-training smoke must confirm that the adapted model produces a parsed tool call within its evaluation budget.

## Observed failure decomposition

| Measure | Observation |
|---|---:|
| Evaluated episodes | 530 |
| Episodes containing reasoning-only/no-tool output | 529 |
| Episodes ending at maximum consecutive rejections | 529 |
| Successful episodes | 1 |
| Rejected proposals | 2,488 |
| Reasoning-only/no-tool rejections | 1,775 (71.3%) |
| Opening-protocol rejections | 678 (27.3%) |
| Other rejections | 35 (1.4%) |

The opening-protocol total includes 490 direct phase violations and 188 cases that incorrectly mixed opening observations with handshake actions in the same atomic cycle.

## Why the dominant error occurred

The evaluation enabled native Thinking and allowed at most 2,048 generated tokens per call. For reasoning-only failures, the saved reasoning had a median serialized length of approximately 9,426 characters and commonly ended mid-sentence. vLLM returned a populated reasoning field but neither response text nor a parsed tool call. The harness therefore correctly reported that there was no action to execute.

This conclusion is high confidence but not yet a controlled causal test because the current results do not persist `finish_reason` or per-call completion-token usage. A smoke comparing the same episodes at 2,048 and a larger budget is required to measure how often additional budget recovers a tool call.

## Parser evidence

| Check | Finding |
|---|---|
| vLLM tool parser | Qwen3 XML parser was configured |
| vLLM reasoning parser | Qwen3 reasoning parser was configured |
| Legal/executed proposals | 721 steps were accepted, proving that the path can return and parse actions |
| Full successful episodes | One episode reached the FSM goal |
| Explicit unparseable-output errors | 31 proposals, about 1.2% of all rejected proposals |

Thus, parser formatting may still account for a small minority of failures, but it does not explain the near-zero aggregate success rate.

## Impact on training

Thinking+rationale training uses structured, non-pretokenized examples and the Qwen Thinking processor at training time. The targets are much shorter than the out-of-box model's generated reasoning:

| Dataset | Rows | Median target length | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| 43/10, 30 trajectories/task | 41,447 | 283 characters | 374 | 966 |
| 47/6, 30 trajectories/task | 45,513 | 283 characters | 373 | 966 |

Teacher-forced training does not call vLLM and does not parse generated actions, so the evaluation failure does not itself invalidate training. The remaining training risk is sequence truncation under the 8,192-token training limit; the trainer rejects samples with zero supervised target tokens, but an explicit Thinking-processor length audit would provide stronger coverage before scaling beyond the smoke.

## Required safeguards before full Thinking evaluation

1. Run a fixed-episode A/B smoke with native Thinking at 2,048 versus a larger completion budget.
2. Persist `finish_reason`, prompt tokens, completion tokens, reasoning length, and whether a tool call was present for every proposal.
3. Require the post-SFT smoke to demonstrate parsed tool calls, not merely a successful vLLM server startup.
4. Evaluate both trained-task and held-task splits only after that smoke passes.
5. Keep reasoning private and excluded from subsequent agent context, as in the current evaluator.
6. Report reasoning-budget exhaustion separately from malformed or invalid tool calls; it is not a model tool-selection error.

## Source scope

This diagnostic uses the aggregate and per-trajectory outputs from jobs `470756` and `470757`, their persisted parallel-run arguments, the current vLLM policy implementation, and the 43/10 and 47/6 rationale training artifacts. It does not infer capability from native simulator success because the model almost never produced enough executable actions to make that metric meaningful.
