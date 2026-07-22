from __future__ import annotations

import unittest

from training.bc_task_vlm.evaluation import (
    VisionGenerationCollator,
    metrics_from_prediction_records,
    score_structured_prediction,
)
from training.bc_task_vlm.prompting import build_user_prompt
from training.bc_task_vlm.schema_utils import (
    TASK_COMPLETE_TOOL_NAME,
    augment_tool_specs_for_agent_prediction,
)
from training.bc_task_vlm.tool_calling import (
    build_tool_schemas,
    parse_first_qwen_tool_call,
    pop_agent_argument,
)

COMM_SPEC = {
    "communicate": {
        "tool_args": ["to", "message"],
        "tool_arg_types": {"to": "STRING", "message": "STRING"},
    }
}


class _RecordingProcessor:
    chat_template = "unused"

    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": [[1]], "attention_mask": [[1]]}


class VisionGenerationCollatorTests(unittest.TestCase):
    def test_text_only_batch_omits_images_processor_argument(self):
        collator = object.__new__(VisionGenerationCollator)
        collator.processor = _RecordingProcessor()
        collator.max_length = None
        collator.sft_format = "tool_call"
        collator.enable_thinking = False
        feature = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "s"}]},
                {"role": "user", "content": [{"type": "text", "text": "u"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            ],
            "image_paths": [],
            "tool_schemas": [],
            **_metadata(),
        }

        collator([feature])

        self.assertNotIn("images", collator.processor.kwargs)


def _metadata(
    *,
    tool: str = "communicate",
    args: dict | None = None,
    agent: str = "agent_0",
    allowed_tool_specs: dict | None = None,
) -> dict:
    args = (
        args
        if args is not None
        else {"to": "agent_1", "message": "agent_0 will move drink_0."}
    )
    return {
        "sample_id": "sample-0",
        "task_name": "beverage_organization",
        "trajectory_id": "traj-0",
        "step_index": 2,
        "agent_id": agent,
        "target_payload": {
            "steps": [{"step": 2, "agent": agent, "tool": tool, "args": args}]
        },
        "target_tool_call": {"name": tool, "arguments": args},
        "target_text": "unused",
        "allowed_tool_specs": (
            allowed_tool_specs if allowed_tool_specs is not None else dict(COMM_SPEC)
        ),
    }


class StructuredPredictionParsingTests(unittest.TestCase):
    def test_tool_call_format_accepts_plain_forced_json(self):
        arguments = {
            "to": "agent_1",
            "message": "agent_0 will move drink_0.",
        }
        metadata = {
            "sample_id": "sample-0",
            "task_name": "beverage_organization",
            "trajectory_id": "traj-0",
            "step_index": 2,
            "agent_id": "agent_0",
            "target_payload": {
                "steps": [
                    {
                        "step": 2,
                        "agent": "agent_0",
                        "tool": "communicate",
                        "args": arguments,
                    }
                ]
            },
            "target_tool_call": {
                "name": "communicate",
                "arguments": arguments,
            },
            "target_text": "unused",
            "allowed_tool_specs": {
                "communicate": {
                    "tool_args": ["to", "message"],
                    "tool_arg_types": {"to": "STRING", "message": "STRING"},
                }
            },
        }

        record = score_structured_prediction(
            decoded_text=(
                '{"tool":"communicate","args":{"to":"agent_1",'
                '"message":"agent_0 will move drink_0."}}'
            ),
            metadata=metadata,
            sft_format="tool_call",
        )

        self.assertIsNone(record["parse_error"])
        self.assertIsNone(record["validation_error"])
        self.assertEqual(record["parsed_tool_call"], metadata["target_tool_call"])
        self.assertTrue(record["exact_tool_call_match"])


class AgentPredictionSchemaTests(unittest.TestCase):
    def test_agent_param_leads_every_tool_schema(self):
        schemas = build_tool_schemas(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs=augment_tool_specs_for_agent_prediction(COMM_SPEC),
            include_agent_param=True,
        )
        names = {schema["function"]["name"] for schema in schemas}
        self.assertIn(TASK_COMPLETE_TOOL_NAME, names)
        for schema in schemas:
            parameters = schema["function"]["parameters"]
            self.assertEqual(next(iter(parameters["properties"])), "agent")
            self.assertEqual(parameters["required"][0], "agent")
            self.assertEqual(
                parameters["properties"]["agent"]["enum"], ["agent_0", "agent_1"]
            )

    def test_v1_schemas_unchanged_without_flag(self):
        schemas = build_tool_schemas(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs=COMM_SPEC,
        )
        parameters = schemas[0]["function"]["parameters"]
        self.assertNotIn("agent", parameters["properties"])
        self.assertEqual(parameters["required"], ["to", "message"])

    def test_pop_agent_argument_roundtrip(self):
        text = (
            '<tool_call>\n{"name": "communicate", "arguments": '
            '{"agent": "agent_1", "to": "agent_0", "message": "done"}}\n</tool_call>'
        )
        agent, stripped = pop_agent_argument(parse_first_qwen_tool_call(text))
        self.assertEqual(agent, "agent_1")
        self.assertEqual(
            stripped["arguments"], {"to": "agent_0", "message": "done"}
        )


class AgentPredictionScoringTests(unittest.TestCase):
    def test_correct_agent_scores_all_matches(self):
        record = score_structured_prediction(
            decoded_text=(
                '<tool_call>\n{"name": "communicate", "arguments": '
                '{"agent": "agent_0", "to": "agent_1", '
                '"message": "agent_0 will move drink_0."}}\n</tool_call>'
            ),
            metadata=_metadata(),
            sft_format="tool_call",
            predict_agent=True,
        )
        self.assertIsNone(record["validation_error"])
        self.assertEqual(record["predicted_agent"], "agent_0")
        self.assertTrue(record["exact_agent_match"])
        self.assertTrue(record["exact_tool_call_match"])
        self.assertTrue(record["exact_action_step_match"])
        # Canonical parsed arguments stay agent-free for the comm judge.
        self.assertNotIn("agent", record["parsed_tool_call"]["arguments"])

    def test_wrong_agent_keeps_tool_args_match(self):
        record = score_structured_prediction(
            decoded_text=(
                '<tool_call>\n{"name": "communicate", "arguments": '
                '{"agent": "agent_1", "to": "agent_1", '
                '"message": "agent_0 will move drink_0."}}\n</tool_call>'
            ),
            metadata=_metadata(),
            sft_format="tool_call",
            predict_agent=True,
        )
        self.assertFalse(record["exact_agent_match"])
        self.assertTrue(record["exact_tool_call_match"])
        self.assertFalse(record["exact_action_step_match"])

    def test_missing_agent_is_validation_error(self):
        record = score_structured_prediction(
            decoded_text=(
                '<tool_call>\n{"name": "communicate", "arguments": '
                '{"to": "agent_1", "message": "x"}}\n</tool_call>'
            ),
            metadata=_metadata(),
            sft_format="tool_call",
            predict_agent=True,
        )
        self.assertIsNotNone(record["validation_error"])
        self.assertIn("agent", record["validation_error"])

    def test_task_complete_and_completion_metrics(self):
        specs = augment_tool_specs_for_agent_prediction(COMM_SPEC)
        terminal = score_structured_prediction(
            decoded_text=(
                '<tool_call>\n{"name": "task_complete", "arguments": '
                '{"agent": "agent_1"}}\n</tool_call>'
            ),
            metadata=_metadata(
                tool=TASK_COMPLETE_TOOL_NAME,
                args={},
                agent="agent_1",
                allowed_tool_specs=specs,
            ),
            sft_format="tool_call",
            predict_agent=True,
        )
        self.assertIsNone(terminal["validation_error"])
        self.assertTrue(terminal["exact_tool_call_match"])
        # Deliberately the wrong agent too (target agent is agent_0), so
        # agent accuracy over the two records is 0.5.
        premature = score_structured_prediction(
            decoded_text=(
                '<tool_call>\n{"name": "task_complete", "arguments": '
                '{"agent": "agent_1"}}\n</tool_call>'
            ),
            metadata=_metadata(allowed_tool_specs=specs),
            sft_format="tool_call",
            predict_agent=True,
        )
        metrics = metrics_from_prediction_records([terminal, premature])
        self.assertEqual(metrics["structured_eval_completion_recall"], 1.0)
        self.assertEqual(metrics["structured_eval_premature_completion_rate"], 1.0)
        self.assertEqual(metrics["structured_eval_agent_accuracy"], 0.5)


class AgentPredictionPromptTests(unittest.TestCase):
    def _prompt(self, predict_agent: bool) -> str:
        return build_user_prompt(
            composite_task="BeverageOrganization",
            task_instruction="Move drinks.",
            agent_id="agent_0",
            next_step_index=2,
            observation_views=["top_view"],
            history_steps=[],
            allowed_tool_specs=COMM_SPEC,
            sft_format="tool_call",
            predict_agent=predict_agent,
        )

    def test_v1_prompt_names_the_acting_agent(self):
        prompt = self._prompt(False)
        self.assertIn("Current acting agent: agent_0", prompt)
        self.assertIn("The acting agent is fixed by the prompt", prompt)

    def test_v2_prompt_asks_model_to_choose(self):
        prompt = self._prompt(True)
        self.assertNotIn("Current acting agent", prompt)
        self.assertIn('pass it as the "agent" argument', prompt)
        self.assertIn("task_complete", prompt)


if __name__ == "__main__":
    unittest.main()
