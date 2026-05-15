"""Metric helpers for task-level BC VLM training and evaluation."""

from __future__ import annotations

import math
from numbers import Real

import torch


def add_perplexity_metrics(metrics: dict[str, object]) -> dict[str, object]:
    """Adds perplexity mirrors for finite scalar loss metrics."""

    updated = dict(metrics)
    for loss_key, perplexity_key in (
        ("loss", "perplexity"),
        ("eval_loss", "eval_perplexity"),
        ("train_loss", "train_perplexity"),
    ):
        if perplexity_key in updated:
            continue
        loss_value = updated.get(loss_key)
        if not isinstance(loss_value, Real) or isinstance(loss_value, bool):
            continue
        if not math.isfinite(float(loss_value)):
            continue
        updated[perplexity_key] = math.exp(float(loss_value))
    return updated


def compute_masked_causal_lm_token_counts(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    logits_to_keep: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Counts argmax token matches only where labels are not masked."""

    if logits_to_keep is not None:
        target_positions = logits_to_keep.to(labels.device) + 1
        valid_position_mask = target_positions < labels.shape[1]
        if not bool(valid_position_mask.all()):
            selected_logit_positions = torch.nonzero(
                valid_position_mask,
                as_tuple=False,
            ).flatten()
            logits = logits.index_select(
                dim=1,
                index=selected_logit_positions.to(logits.device),
            )
            target_positions = target_positions.index_select(
                dim=0,
                index=selected_logit_positions.to(target_positions.device),
            )
        labels = labels.index_select(dim=1, index=target_positions)
    else:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
    predictions = logits.argmax(dim=-1)
    label_mask = labels != -100
    if label_mask.numel() == 0:
        return 0, 0
    correct = ((predictions == labels) & label_mask).sum().item()
    total = label_mask.sum().item()
    return int(correct), int(total)


def build_token_accuracy_metrics(
    *,
    correct_tokens: int,
    total_tokens: int,
    metric_key_prefix: str,
) -> dict[str, float]:
    """Builds token accuracy metrics with explicit numerator and denominator."""

    accuracy = correct_tokens / total_tokens if total_tokens else 0.0
    return {
        f"{metric_key_prefix}_token_accuracy": float(accuracy),
        f"{metric_key_prefix}_token_correct_count": float(correct_tokens),
        f"{metric_key_prefix}_token_count": float(total_tokens),
    }


def build_structured_eval_metrics(
    *,
    total_samples: int,
    parsed_tool_calls: int,
    valid_tool_calls: int,
    exact_tool_matches: int,
    exact_args_matches: int,
    exact_tool_call_matches: int,
    exact_action_matches: int,
) -> dict[str, float]:
    """Builds structured generation metrics for one-tool-call validation."""

    def rate(count: int) -> float:
        if total_samples == 0:
            return 0.0
        return count / total_samples

    return {
        "structured_eval_num_samples": float(total_samples),
        "structured_eval_target_tool_call_count": float(total_samples),
        "structured_eval_parsed_tool_call_count": float(parsed_tool_calls),
        "structured_eval_valid_tool_call_count": float(valid_tool_calls),
        "structured_eval_exact_tool_match_count": float(exact_tool_matches),
        "structured_eval_exact_args_match_count": float(exact_args_matches),
        "structured_eval_exact_tool_call_match_count": float(exact_tool_call_matches),
        "structured_eval_exact_action_step_match_count": float(exact_action_matches),
        "structured_eval_tool_call_parse_rate": float(rate(parsed_tool_calls)),
        "structured_eval_tool_call_valid_rate": float(rate(valid_tool_calls)),
        "structured_eval_exact_tool_accuracy": float(rate(exact_tool_matches)),
        "structured_eval_exact_args_match_rate": float(rate(exact_args_matches)),
        "structured_eval_exact_tool_call_match_rate": float(
            rate(exact_tool_call_matches)
        ),
        "structured_eval_tool_call_accuracy": float(rate(exact_tool_call_matches)),
        "structured_eval_exact_action_step_match_rate": float(
            rate(exact_action_matches)
        ),
        "structured_eval_action_prediction_accuracy": float(rate(exact_action_matches)),
    }
