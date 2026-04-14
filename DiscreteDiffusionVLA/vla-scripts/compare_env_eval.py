"""
Compare LiberoRLEnv obs vs eval's prepare_observation obs, side-by-side.

Uses the same LIBERO task + init_state and prints exact tensor stats
for input_ids, pixel_values, proprio to identify any mismatch.
"""
import json
import os
import sys
import numpy as np
import torch
from dataclasses import dataclass
import draccus

from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

AutoConfig.register("openvla", OpenVLAConfig)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


@dataclass
class Cfg:
    vla_path: str = ""
    task_id: int = 0
    task_suite: str = "libero_object"
    dataset_name: str = "libero_object_no_noops"


def describe(name, t):
    if t is None:
        print(f"  {name}: None")
        return
    if isinstance(t, np.ndarray):
        arr = t
    else:
        arr = t.detach().cpu().numpy()
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.4f} max={arr.max():.4f} "
          f"mean={arr.astype(np.float64).mean():.4f}")


@draccus.wrap()
def main(cfg: Cfg):
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

    os.environ.setdefault("MUJOCO_GL", "osmesa")

    # Load processor + stats
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    )
    stats_path = os.path.join(cfg.vla_path, "dataset_statistics.json")
    if os.path.exists(stats_path):
        vla.norm_stats.update(json.load(open(stats_path)))

    proprio_stats = vla.norm_stats[cfg.dataset_name]["proprio"]

    # ---- LIBERO env raw obs ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[cfg.task_suite]()
    task = suite.get_task(cfg.task_id)
    init_states = suite.get_task_init_states(cfg.task_id)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    raw_obs = env.set_init_state(init_states[0])
    # Run 10 dummy steps
    for _ in range(10):
        raw_obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    print(f"\n=== Raw LIBERO obs (task {cfg.task_id}: {task.language[:50]}) ===")
    print(f"  agentview_image shape: {raw_obs['agentview_image'].shape}, range: {raw_obs['agentview_image'].min()}-{raw_obs['agentview_image'].max()}")
    print(f"  wrist_image shape: {raw_obs['robot0_eye_in_hand_image'].shape}")
    print(f"  eef_pos: {raw_obs['robot0_eef_pos']}")
    print(f"  eef_quat: {raw_obs['robot0_eef_quat']}")
    print(f"  gripper_qpos: {raw_obs['robot0_gripper_qpos']}")

    # ---- EVAL pipeline ----
    print("\n=== EVAL pipeline (prepare_observation + get_vla_action) ===")
    sys.path.insert(0, "/home/ubuntu/VLA-DFM/DiscreteDiffusionVLA/experiments/robot")
    from robot.libero.run_libero_eval import prepare_observation
    from robot.openvla_utils import normalize_proprio, prepare_images_for_vla
    from types import SimpleNamespace

    eval_obs, _ = prepare_observation(raw_obs, resize_size=(224, 224))
    print(f"  Prepared observation keys: {list(eval_obs.keys())}")
    print(f"  full_image: shape={eval_obs['full_image'].shape} min={eval_obs['full_image'].min()} max={eval_obs['full_image'].max()}")
    print(f"  wrist_image: shape={eval_obs['wrist_image'].shape}")
    print(f"  state (raw): {eval_obs['state']}")

    # Proprio normalization
    eval_proprio = normalize_proprio(eval_obs["state"], proprio_stats)
    print(f"  state (normalized): {eval_proprio}")

    # Image preprocessing
    cfg_ns = SimpleNamespace(center_crop=True)
    all_images = [eval_obs["full_image"], eval_obs["wrist_image"]]
    all_imgs_processed = prepare_images_for_vla(all_images, cfg_ns)
    print(f"  After prepare_images_for_vla: primary shape={all_imgs_processed[0].size if hasattr(all_imgs_processed[0], 'size') else all_imgs_processed[0].shape}")

    # Processor call (same as eval)
    from prismatic.vla.prompt_utils import build_vla_prompt
    prompt = build_vla_prompt(task.language, legacy=True)
    primary_pil = all_imgs_processed[0]
    eval_inputs = processor(prompt, primary_pil)
    wrist_inputs = processor(prompt, all_imgs_processed[1])
    eval_pv = torch.cat([eval_inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

    print(f"\n  EVAL input_ids: {eval_inputs['input_ids'].shape}")
    describe("EVAL pixel_values", eval_pv)
    describe("EVAL proprio (normalized)", eval_proprio)

    # ---- OUR ENV ----
    print("\n=== OUR LiberoRLEnv obs ===")
    from prismatic.rl.libero_env import LiberoRLEnv
    our_env = LiberoRLEnv(
        task_suite=cfg.task_suite, task_id=cfg.task_id,
        processor=processor, vla_config=vla.config,
        unnorm_key=cfg.dataset_name,
        proprio_norm_stats=proprio_stats,
    )
    our_obs = our_env.reset()
    print(f"  input_ids: {our_obs['input_ids'].shape}")
    describe("OUR pixel_values", our_obs["pixel_values"])
    describe("OUR proprio", our_obs["proprio"])
    print(f"  input_ids match: {torch.equal(eval_inputs['input_ids'][0], our_obs['input_ids'][0, :eval_inputs['input_ids'].shape[1]])}")

    # Pixel diff
    if eval_pv.shape == our_obs["pixel_values"].shape:
        diff = (eval_pv.float() - our_obs["pixel_values"].float()).abs()
        print(f"\n  Pixel diff: max={diff.max().item():.4f} mean={diff.mean().item():.6f}")
    else:
        print(f"\n  PIXEL SHAPE MISMATCH: eval={eval_pv.shape} ours={our_obs['pixel_values'].shape}")


if __name__ == "__main__":
    main()
