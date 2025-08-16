import torch

from hoidini.object_conditioning.object_pointcloud_dataset import pyg_collate_wrapper


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["inp"] for b in notnone_batches]
    if "lengths" in notnone_batches[0]:
        lenbatch = [b["lengths"] for b in notnone_batches]
    else:
        lenbatch = [len(b["inp"][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = (
        lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1])
        .unsqueeze(1)
        .unsqueeze(1)
    )  # unsqueeze for broadcasting

    motion = databatchTensor
    cond = {"y": {"mask": maskbatchTensor, "lengths": lenbatchTensor}}

    if "text" in notnone_batches[0]:
        textbatch = [b["text"] for b in notnone_batches]
        cond["y"].update({"text": textbatch})

    if "tokens" in notnone_batches[0]:
        textbatch = [b["tokens"] for b in notnone_batches]
        cond["y"].update({"tokens": textbatch})

    if "action" in notnone_batches[0]:
        actionbatch = [b["action"] for b in notnone_batches]
        cond["y"].update({"action": torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if "action_text" in notnone_batches[0]:
        action_text = [b["action_text"] for b in notnone_batches]
        cond["y"].update({"action_text": action_text})

    if "prefix" in notnone_batches[0]:
        cond["y"].update(
            {"prefix": collate_tensors([b["prefix"] for b in notnone_batches])}
        )

    if "key" in notnone_batches[0]:
        cond["y"].update({"db_key": [b["key"] for b in notnone_batches]})

    if "obj_points" in notnone_batches[0]:
        cond["y"].update(
            {
                "obj_points": pyg_collate_wrapper(
                    [b["obj_points"] for b in notnone_batches]
                )
            }
        )

    if "metadata" in notnone_batches[0]:
        cond["y"].update({"metadata": [b["metadata"] for b in notnone_batches]})

    if "condition_mask" in notnone_batches[0]:
        cond["y"].update(
            {"condition_mask": [b["condition_mask"] for b in notnone_batches]}
        )

    if "condition_input" in notnone_batches[0]:
        cond["y"].update(
            {"condition_input": [b["condition_input"] for b in notnone_batches]}
        )

    if "tfms_root_global" in notnone_batches[0]:
        cond["y"].update(
            {
                "tfms_root_global": collate_tensors(
                    [b["tfms_root_global"] for b in notnone_batches]
                )
            }
        )

    if "tfm_processor" in notnone_batches[0]:
        cond["y"].update(
            {
                "tfm_processor": collate_tensors(
                    [b["tfm_processor"] for b in notnone_batches]
                )
            }
        )

    if "is_zero_hoi_mask" in notnone_batches[0]:
        cond["y"].update(
            {
                "is_zero_hoi_mask": torch.tensor(
                    [e["is_zero_hoi_mask"] for e in notnone_batches]
                )
            }
        )

    if "loss_mask" in notnone_batches[0]:
        cond["y"].update(
            {"loss_mask": collate_tensors([e["loss_mask"] for e in notnone_batches])}
        )  # (B, n_features)

    return motion, cond


# an adapter to our collate func
def t2m_collate(batch, target_batch_size):
    repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
    repeated_batch = batch * repeat_factor
    full_batch = repeated_batch[:target_batch_size]  # Truncate to the target batch size
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [
        {
            "inp": torch.tensor(b[4].T)
            .float()
            .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            "text": b[2],  # b[0]['caption']
            "tokens": b[6],
            "lengths": b[5],
            "key": b[7] if len(b) > 7 else None,
        }
        for b in full_batch
    ]
    return collate(adapted_batch)


def t2m_prefix_collate(batch, pred_len):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [
        {
            "inp": torch.tensor(b[4].T)
            .float()
            .unsqueeze(1)[..., -pred_len:],  # [seqlen, J] -> [J, 1, seqlen]
            "prefix": torch.tensor(b[4].T).float().unsqueeze(1)[..., :-pred_len],
            "text": b[2],  # b[0]['caption']
            "tokens": b[6],
            "lengths": pred_len,  # b[5],
            "key": b[7] if len(b) > 7 else None,
        }
        for b in batch
    ]
    return collate(adapted_batch)
