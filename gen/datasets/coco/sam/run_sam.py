import autoroot

import math

import fire
import segment_anything_fast
import torch
import tqdm
from gen.datasets.coco.sam.save_custom_json import save_json, update_with_inference_batch
from gen.datasets.coco.sam.data import build_data, setup_coco_img_ids


def unbind_jagged(device, data, sizes, offsets):
    if data is None:
        return None
    data = data.to(device=device, non_blocking=True)
    return [data[offsets[batch_idx] : offsets[batch_idx + 1]].view(sizes[batch_idx]) for batch_idx in range(len(sizes))]


PADDED_TENSOR = None


# Preallocate a "landing" Tensor for incoming data and reuse it across launches.
def pad_to_batch_size(batch, batch_size, device):
    assert batch.dim() == 4
    # assert batch.is_pinned()
    global PADDED_TENSOR
    if PADDED_TENSOR is None:
        batch = batch.to(device=device, non_blocking=True)
        full_batch_size = (batch_size, batch.size(1), batch.size(2), batch.size(3))
        first_entry = batch[0].unsqueeze(0)
        repeat_first_entry = first_entry.expand(full_batch_size)
        padded_batch = torch.cat([batch, repeat_first_entry[batch.size(0) : batch_size]], dim=0)
        assert padded_batch.size() == full_batch_size
        PADDED_TENSOR = padded_batch
    PADDED_TENSOR[: batch.size(0)].copy_(batch, non_blocking=True)
    return PADDED_TENSOR


def create_result_entry_inference(anns, gt_masks_list, masks, scores, img_idx):
    argmax_scores = torch.argmax(scores, dim=1)
    inference_masks = masks.gather(
        1, argmax_scores.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((masks.size(0), 1, masks.size(2), masks.size(3)))
    ).squeeze(1)

    return anns[0]["image_id"], inference_masks


def build_results_batch_nested(predictor, batch, batch_size, pad_input_image_batch, annotations, annotation_id):
    encoder = predictor.model.image_encoder
    device = predictor.device

    input_image_batch = batch[0]
    if input_image_batch is not None:
        # The number of valid data points varies slightly per batch
        orig_input_image_batch_size = input_image_batch.size(0)

    if input_image_batch is None:
        return (None, None, None, annotations, annotation_id)
    if batch[1] is None:
        return (None, None, None, annotations, annotation_id)

    with torch.autograd.profiler.record_function("nt data transfer"):
        datapoints = list(zip(*(batch[7:])))
        nt_coords = batch[1].to(device=device, non_blocking=True)
        gt_masks_lists = unbind_jagged(*([device] + batch[4:7]))
        nt_fg_labels = torch.ones_like(nt_coords, dtype=torch.int).prod(dim=-1, keepdim=True)
        if pad_input_image_batch:
            # Pad to a static shape to avoid recompilation
            input_image_batch = pad_to_batch_size(input_image_batch, batch_size, device)
        else:
            input_image_batch = input_image_batch.to(device=device, non_blocking=True)

    # We explicitly exclude data transfers from the timing to focus
    # only on the kernel performance.
    # Next we synchronize and set two events to start timing.
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with torch.autograd.profiler.record_function("timed region"):
        with torch.autograd.profiler.record_function("image encoder"):
            features_batch = encoder(input_image_batch)
            features_batch = features_batch[:orig_input_image_batch_size]

        with torch.autograd.profiler.record_function("nt predict_torch"):
            predictor.reset_image()
            predictor.original_sizes = [d[1].shape[:2] for d in datapoints]
            predictor.input_sizes = [d[2] for d in datapoints]
            predictor.features = features_batch
            predictor.is_image_set = True
            nt_coords = nt_coords.unsqueeze(2)
            masks, scores, logits = predictor.predict_torch(
                point_coords=nt_coords,
                point_labels=nt_fg_labels,
                multimask_output=True,
            )
            if annotations is not None:
                inference_batch = [
                    create_result_entry_inference(d[0], g, m, s, d[3])
                    for (m, s, d, g) in zip(masks.unbind(), scores.unbind(), datapoints, gt_masks_lists)
                ]
                annotations, annotation_id = update_with_inference_batch(inference_batch, annotations, annotation_id, predictor.original_sizes)
        # After all kernels have been launched we synchronize again and measure
        # the amount of time spent on the GPU. This is a fairly tight measurement
        # around the launched GPU kernels and excludes data movement from host
        # to device.
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
    return None, orig_input_image_batch_size, elapsed_time, annotations, annotation_id


def build_results(
    batched_data_iter,
    predictor,
    mask_debug_out_dir,
    batch_size,
    use_compile,
    use_compile_decoder,
    use_nested_tensor,
    pad_input_image_batch,
    coco_root_dir,
    coco_slice_name,
    split_idx,
    use_fullgraph=False,
):

    # TODO: Re-enable this for datapoints
    assert not use_compile_decoder

    batch_runner = None
    if use_nested_tensor:
        batch_runner = build_results_batch_nested

    results = []
    batch_idx = 0
    num_images = 0
    num_batches = 0
    annotation_id = 0
    annotations = []
    for batch in tqdm.tqdm(batched_data_iter):
        with torch.no_grad():
            if batch_idx == 0:
                with torch.autograd.profiler.record_function("compilation and warmup"):
                    if str(use_compile) != "False":
                        predictor.model.image_encoder = torch.compile(predictor.model.image_encoder, mode=use_compile, fullgraph=use_fullgraph)
                    # Run first batch a few times for warmup and exclude it from the final timings
                    for _ in range(3):
                        _ = batch_runner(predictor, batch, batch_size, pad_input_image_batch, None, None)
            result_batch, num_datapoints, kernel_time, annotations, annotation_id = batch_runner(
                predictor, batch, batch_size, pad_input_image_batch, annotations=annotations, annotation_id=annotation_id
            )
            if result_batch is not None:
                results += result_batch

        batch_idx += 1

        if batch_idx % 1000 == 0:
            print(f"Saving annotations for batch {batch_idx} of {len(batched_data_iter)}")
            save_json(coco_root_dir, coco_slice_name, annotations, split_idx)

    save_json(coco_root_dir, coco_slice_name, annotations, split_idx)

    return results, num_batches, num_images, annotations


def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

def identity_runner(fn, *args, **kwargs):
    return fn(*args, **kwargs)

def run(
    coco_root_dir,
    coco_slice_name,
    sam_checkpoint_base_path,
    sam_model_type,
    point_sampling_cache_dir,
    mask_debug_out_dir,
    batch_size=1,
    print_header=False,
    coco_category_names=None,
    limit=None,
    img_id=None,
    use_half=None,
    use_compile="False",
    use_compile_decoder=False,
    compress=None,
    epilogue_fusion_first=False,
    num_workers=0,
    use_nested_tensor=False,
    use_rel_pos=True,
    pad_input_image_batch=True,
    profile_path=None,
    profile_top=False,
    memory_path=None,
    use_local_sam_fork=False,
    use_compiler_settings=False,
    split_size=1,
    split_idx=0,
):
    torch._dynamo.config.cache_size_limit = 50000
    from torch._inductor import config as inductorconfig

    inductorconfig.triton.unique_kernel_names = True
    inductorconfig.epilogue_fusion_first = epilogue_fusion_first

    if use_compiler_settings:
        # inductorconfig.fx_graph_cache = True # seems to slow performance
        inductorconfig.epilogue_fusion = False
        inductorconfig.coordinate_descent_tuning = True
        inductorconfig.coordinate_descent_check_all_directions = True

    if use_half is not None:
        if use_half == "float16":
            use_half = torch.float16
        elif use_half == "bfloat16":
            use_half = torch.bfloat16
        else:
            raise ValueError("Expected one of float16 or bfloat for specified {use_half}")

    # Batch size needs to be a multiple of two and at most 512.
    assert math.log2(batch_size).is_integer()
    assert batch_size <= 512

    # https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints
    # largest to smallest: vit_h, vit_l, vit_b
    model_type_to_checkpoint = {
        "vit_h": f"{sam_checkpoint_base_path}/sam_vit_h_4b8939.pth",
        "vit_l": f"{sam_checkpoint_base_path}/sam_vit_l_0b3195.pth",
        "vit_b": f"{sam_checkpoint_base_path}/sam_vit_b_01ec64.pth",
    }

    if use_local_sam_fork:
        from segment_anything_fast import SamPredictor, sam_model_registry
    else:
        from segment_anything import SamPredictor, sam_model_registry
    checkpoint_path = model_type_to_checkpoint[sam_model_type]
    sam = sam_model_registry[sam_model_type](checkpoint=checkpoint_path).cuda()
    predictor = SamPredictor(sam)

    from segment_anything_fast import tools

    tools.apply_eval_dtype_predictor(predictor, use_half)

    for block in predictor.model.image_encoder.blocks:
        block.attn.use_rel_pos = use_rel_pos

    coco_img_ids, cat_id_to_cat, catIds, coco = setup_coco_img_ids(coco_root_dir, coco_slice_name, coco_category_names, img_id)

    build_batch = build_data(
        coco_img_ids,
        coco,
        catIds,
        coco_root_dir,
        coco_slice_name,
        point_sampling_cache_dir,
        predictor,
        use_half,
        use_nested_tensor,
        pad_input_image_batch,
    )

    limit = len(coco_img_ids) if limit is None else limit
    all_idx = split_range(range(limit), split_size)
    idx = list(all_idx)[split_idx]
    batched_data_iter = torch.utils.data.DataLoader(idx, batch_size=batch_size, collate_fn=build_batch, num_workers=num_workers, pin_memory=False)
    runner = identity_runner

    results, num_batches, num_images, annotations = runner(
        build_results,
        batched_data_iter,
        predictor,
        mask_debug_out_dir,
        batch_size,
        use_compile,
        use_compile_decoder,
        use_nested_tensor,
        pad_input_image_batch,
        coco_root_dir,
        coco_slice_name,
        split_idx,
    )


if __name__ == "__main__":
    fire.Fire(run)
