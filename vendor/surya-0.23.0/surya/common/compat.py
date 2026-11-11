from typing import Optional, Iterable, List, Union, Tuple

import torch


def verify_out_features_out_indices(
    out_features: Optional[Iterable[str]],
    out_indices: Optional[Iterable[int]],
    stage_names: Optional[Iterable[str]],
):
    """
    Verify that out_indices and out_features are valid for the given stage_names.
    """
    if stage_names is None:
        raise ValueError("Stage_names must be set for transformers backbones")

    if out_features is not None:
        if not isinstance(out_features, (list,)):
            raise ValueError(f"out_features must be a list got {type(out_features)}")
        if any(feat not in stage_names for feat in out_features):
            raise ValueError(
                f"out_features must be a subset of stage_names: {stage_names} got {out_features}"
            )
        if len(out_features) != len(set(out_features)):
            raise ValueError(
                f"out_features must not contain any duplicates, got {out_features}"
            )
        if out_features != (
            sorted_feats := [feat for feat in stage_names if feat in out_features]
        ):
            raise ValueError(
                f"out_features must be in the same order as stage_names, expected {sorted_feats} got {out_features}"
            )

    if out_indices is not None:
        if not isinstance(out_indices, (list, tuple)):
            raise ValueError(
                f"out_indices must be a list or tuple, got {type(out_indices)}"
            )
        # Convert negative indices to their positive equivalent: [-1,] -> [len(stage_names) - 1,]
        positive_indices = tuple(
            idx % len(stage_names) if idx < 0 else idx for idx in out_indices
        )
        if any(idx for idx in positive_indices if idx not in range(len(stage_names))):
            raise ValueError(
                f"out_indices must be valid indices for stage_names {stage_names}, got {out_indices}"
            )
        if len(positive_indices) != len(set(positive_indices)):
            msg = f"out_indices must not contain any duplicates, got {out_indices}"
            msg += (
                f"(equivalent to {positive_indices}))"
                if positive_indices != out_indices
                else ""
            )
            raise ValueError(msg)
        if positive_indices != tuple(sorted(positive_indices)):
            sorted_negative = tuple(
                idx
                for _, idx in sorted(
                    zip(positive_indices, out_indices), key=lambda x: x[0]
                )
            )
            raise ValueError(
                f"out_indices must be in the same order as stage_names, expected {sorted_negative} got {out_indices}"
            )

    if out_features is not None and out_indices is not None:
        if len(out_features) != len(out_indices):
            raise ValueError(
                "out_features and out_indices should have the same length if both are set"
            )
        if out_features != [stage_names[idx] for idx in out_indices]:
            raise ValueError(
                "out_features and out_indices should correspond to the same stages if both are set"
            )


def _align_output_features_output_indices(
    out_features: Optional[List[str]],
    out_indices: Optional[Union[List[int], Tuple[int]]],
    stage_names: List[str],
):
    """
    Finds the corresponding `out_features` and `out_indices` for the given `stage_names`.

    The logic is as follows:
        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the
        `out_indices`.
        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the
        `out_features`.
        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.
        - `out_indices` and `out_features` set: input `out_indices` and `out_features` are returned.

    Args:
        out_features (`List[str]`): The names of the features for the backbone to output.
        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.
        stage_names (`List[str]`): The names of the stages of the backbone.
    """
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) - 1]
        out_features = [stage_names[-1]]
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]
    return out_features, out_indices


def get_aligned_output_features_output_indices(
    out_features: Optional[List[str]],
    out_indices: Optional[Union[List[int], Tuple[int]]],
    stage_names: List[str],
) -> Tuple[List[str], List[int]]:
    """
    Get the `out_features` and `out_indices` so that they are aligned.

    The logic is as follows:
        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the
        `out_indices`.
        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the
        `out_features`.
        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.
        - `out_indices` and `out_features` set: they are verified to be aligned.

    Args:
        out_features (`List[str]`): The names of the features for the backbone to output.
        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.
        stage_names (`List[str]`): The names of the stages of the backbone.
    """
    # First verify that the out_features and out_indices are valid
    verify_out_features_out_indices(
        out_features=out_features, out_indices=out_indices, stage_names=stage_names
    )
    output_features, output_indices = _align_output_features_output_indices(
        out_features=out_features, out_indices=out_indices, stage_names=stage_names
    )
    # Verify that the aligned out_features and out_indices are valid
    verify_out_features_out_indices(
        out_features=output_features,
        out_indices=output_indices,
        stage_names=stage_names,
    )
    return output_features, output_indices


def find_pruneable_heads_and_indices(
    heads: list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.LongTensor]:
    mask = torch.ones(n_heads, head_size, dtype=torch.bool)

    heads = set(heads) - already_pruned_heads

    for head in heads:
        # Shift the head index left by however many smaller heads
        # were already removed earlier.
        shifted_head = head - sum(1 for h in already_pruned_heads if h < head)
        mask[shifted_head] = False

    index = torch.arange(n_heads * head_size)[mask.view(-1)].long()
    return heads, index
