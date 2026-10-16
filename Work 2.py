"""
Run UniGEM from PyCharm cells.

Usage:
1. Open this file in PyCharm.
2. Make sure the selected interpreter is your connected server interpreter.
3. Right-click each "# %%" cell and choose "Execute Cell in Console".
4. Run cells from top to bottom.
"""

# %%
# Cell 1: configure server paths.

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/zxy/ligand_diffusion_screening")
UNIGEM_ROOT = PROJECT_ROOT / "UniGEM-master"
QM9_DATADIR = PROJECT_ROOT / "data" / "qm9_data"
MODEL_PATH = UNIGEM_ROOT / "outputs" / "split_k8_t10_with_atom_type_prop_pred_lumo"

os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["UNIGEM_ROOT"] = str(UNIGEM_ROOT)
os.environ["QM9_DATADIR"] = str(QM9_DATADIR)
os.environ["MODEL_PATH"] = str(MODEL_PATH)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("NUM_WORKERS", "4")
os.environ.setdefault("N_SAMPLES", "100")
os.environ.setdefault("CHECKPOINT_EPOCH", "2000")

print("Python:", sys.executable)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("UNIGEM_ROOT:", UNIGEM_ROOT)
print("QM9_DATADIR:", QM9_DATADIR)
print("MODEL_PATH:", MODEL_PATH)


# %%
# Cell 2: check that PyCharm is using the server path and the project exists.

print("Current working directory before change:", Path.cwd())
os.chdir(PROJECT_ROOT)
print("Current working directory after change:", Path.cwd())

required_paths = [
    UNIGEM_ROOT,
    PROJECT_ROOT / "scripts" / "server" / "run_unigem.py",
    PROJECT_ROOT / "scripts" / "server" / "download_qm9_data.py",
]

for path in required_paths:
    print(("[ok] " if path.exists() else "[missing] ") + str(path))


# %%
# Cell 3: check key Python packages on the server interpreter.

for module_name in ["torch", "numpy", "scipy", "tqdm", "wandb", "rdkit", "imageio", "matplotlib"]:
    try:
        module = __import__(module_name)
        print(f"[ok] {module_name}: {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"[missing] {module_name}: {exc}")

try:
    import torch

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA device 0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("CUDA check failed:", exc)


# %%
# Cell 4: install missing lightweight Python packages.
# Run this if Cell 3 reports missing wandb, imageio, or matplotlib.

subprocess.run(
    [sys.executable, "-m", "pip", "install", "wandb", "imageio", "matplotlib"],
    check=True,
)


# %%
# Cell 5: download QM9 raw data to the server.
# Run this once. If files already exist, it will skip them.

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "download_qm9"],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 6: preprocess QM9 into train.npz / valid.npz / test.npz.
# Run this after Cell 5 finishes successfully.

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "prepare_qm9"],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 7: verify QM9 files.

qm9_dir = QM9_DATADIR / "qm9"
for filename in [
    "dsgdb9nsd.xyz.tar.bz2",
    "uncharacterized.txt",
    "atomref.txt",
    "train.npz",
    "valid.npz",
    "test.npz",
]:
    path = qm9_dir / filename
    if path.exists():
        print(f"[ok] {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"[missing] {path}")


# %%
# Cell 8: train QM9 UniGEM LUMO.
# First run this smoke test. It should finish much faster than full training.

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "train_qm9_smoke"],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 9: train a medium QM9 UniGEM LUMO demo.
# This is the recommended next step after the smoke test.

os.environ["MODEL_PATH"] = str(UNIGEM_ROOT / "outputs" / "demo_qm9_unigem_lumo")
os.environ.setdefault("DEMO_EPOCHS", "10")
os.environ.setdefault("DEMO_BATCH_SIZE", "32")

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "train_qm9_demo"],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 10: train full QM9 UniGEM LUMO.
# This is the long full GPU job.

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "train_qm9_lumo"],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 11: evaluate generation.
# Run this only after a checkpoint exists, e.g. generative_model_ema_2000.npy.

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "eval_generation"],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 12: print result locations.

print("Training output directory:")
print(MODEL_PATH)

print("\nTraining log:")
print(UNIGEM_ROOT / "logs" / "train_qm9_unigem_lumo.log")

print("\nEvaluation log:")
print(UNIGEM_ROOT / "logs" / "eval_generation.log")

print("\nGenerated XYZ directory, if SAVE_TO_XYZ=1:")
print(MODEL_PATH / "eval" / "analyzed_molecules")


# %%
# Cell 13: filter generated molecules that RDKit can parse as valid.

subprocess.run(
    [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "server" / "filter_valid_molecules.py"),
        "--input_dir",
        str(UNIGEM_ROOT / "outputs" / "demo_qm9_unigem_lumo" / "eval" / "analyzed_molecules"),
        "--output_dir",
        str(UNIGEM_ROOT / "outputs" / "demo_qm9_unigem_lumo" / "eval" / "valid_molecules"),
    ],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 14: create figures and tables for group-meeting report.

subprocess.run(
    [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "server" / "make_report_figures.py"),
    ],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 15: task-specific filtering with SAMGen chemistry rules and external predictor.
# Adjust ROW_IDX / TARGET_Y / MIN_PRED_Y for your research task.

ROW_IDX = 0
TARGET_Y = None
MIN_PRED_Y = None

cmd = [
    sys.executable,
    str(PROJECT_ROOT / "scripts" / "server" / "unigem_task_filter_with_predictor.py"),
    "--row_idx",
    str(ROW_IDX),
]
if TARGET_Y is not None:
    cmd += ["--target_y", str(TARGET_Y), "--target_window", "0.7"]
if MIN_PRED_Y is not None:
    cmd += ["--min_pred_y", str(MIN_PRED_Y)]

subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


# %%
# Cell 16: configure the task-adapted full UniGEM run.
# Use these cells when you restart from scratch for your own research task.

TASK_EXP_NAME = "task_qm9_unigem_lumo"
TASK_FINETUNE_EXP_NAME = "task_qm9_unigem_lumo_finetuned"
TASK_MODEL_PATH = UNIGEM_ROOT / "outputs" / TASK_EXP_NAME
TASK_BASE_MODEL_PATH = TASK_MODEL_PATH
TASK_FINETUNE_MODEL_PATH = UNIGEM_ROOT / "outputs" / TASK_FINETUNE_EXP_NAME
TASK_QM9_DATADIR = PROJECT_ROOT / "data" / "task_qm9_data"
GENERATION_DATASET = "qm9"

os.environ["FULL_EXP_NAME"] = TASK_EXP_NAME
os.environ["MODEL_PATH"] = str(TASK_MODEL_PATH)
os.environ["TASK_QM9_DATADIR"] = str(TASK_QM9_DATADIR)
os.environ["CHECKPOINT_EPOCH"] = "final"
os.environ["N_SAMPLES"] = "1000"
os.environ["BATCH_SIZE_GEN"] = "100"
os.environ["SAMPLE_STEPS"] = "1000"
os.environ["FULL_EPOCHS"] = "3000"
os.environ["NUM_WORKERS"] = "0"

print("Task experiment:", TASK_EXP_NAME)
print("Task model path:", TASK_MODEL_PATH)
print("Task finetune experiment:", TASK_FINETUNE_EXP_NAME)
print("Task finetune model path:", TASK_FINETUNE_MODEL_PATH)
print("Task domain datadir:", TASK_QM9_DATADIR)
print("Full epochs:", os.environ["FULL_EPOCHS"])


# %%
# Cell 17: train the task-adapted full UniGEM model from scratch.
# This is the normal/full run, not the 50-epoch demo.

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "train_qm9_lumo"],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 17.5: build task-domain dataset and finetune UniGEM on your Pu_logD ligands.
# Run this after Cell 17 finishes and before Cell 18.

os.environ["TASK_QM9_DATADIR"] = str(TASK_QM9_DATADIR)
os.environ["TASK_PRETRAINED_MODEL"] = str(TASK_BASE_MODEL_PATH / "generative_model_ema_final.npy")
os.environ["TASK_FINETUNE_EXP_NAME"] = TASK_FINETUNE_EXP_NAME
os.environ["TASK_FINETUNE_EPOCHS"] = "300"
os.environ["TASK_FINETUNE_BATCH_SIZE"] = "2"
os.environ["TASK_FINETUNE_SAMPLE_BATCH_SIZE"] = "2"
os.environ["TASK_FINETUNE_LR"] = "1e-5"
os.environ["TASK_FINETUNE_NUM_WORKERS"] = "0"
os.environ["TASK_FINETUNE_TEST_EPOCHS"] = "999"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "prepare_task_qm9"],
    cwd=PROJECT_ROOT,
    check=True,
)

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "prepare_task_scaffold_library"],
    cwd=PROJECT_ROOT,
    check=True,
)

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "finetune_qm9_task"],
    cwd=PROJECT_ROOT,
    check=True,
)

TASK_MODEL_PATH = TASK_FINETUNE_MODEL_PATH
GENERATION_DATASET = "qm9_task"
os.environ["MODEL_PATH"] = str(TASK_MODEL_PATH)
os.environ["CHECKPOINT_EPOCH"] = "final"

print("Using finetuned task-domain model for generation:")
print(TASK_MODEL_PATH)


# %%
# Cell 18: generate molecules from the task full checkpoint and keep RDKit-valid molecules.

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "server" / "run_unigem.py"), "--task", "eval_generation"],
    cwd=PROJECT_ROOT,
    check=True,
)

subprocess.run(
    [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "server" / "filter_valid_molecules.py"),
        "--input_dir",
        str(TASK_MODEL_PATH / "eval" / "analyzed_molecules"),
        "--output_dir",
        str(TASK_MODEL_PATH / "eval" / "valid_molecules"),
        "--dataset",
        GENERATION_DATASET,
    ],
    cwd=PROJECT_ROOT,
    check=True,
)


# %%
# Cell 18.5: scaffold-first candidate generation for the Pu_logD task.
# Use this when UniGEM free generation does not produce task-like connected ligands.

SCAFFOLD_FIRST_DIR = TASK_MODEL_PATH / "eval" / "scaffold_first_candidates"

subprocess.run(
    [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "server" / "scaffold_first_candidate_generator.py"),
        "--output_dir",
        str(SCAFFOLD_FIRST_DIR),
        "--max_candidates",
        "3000",
        "--analog_hops",
        "1",
        "--max_analog_jobs_per_seed",
        "80",
        "--max_mw",
        "700",
        "--max_hbd",
        "4",
        "--require_task_scaffold_or_motif",
    ],
    cwd=PROJECT_ROOT,
    check=True,
)

VALID_CSV_FOR_TASK_FILTER = SCAFFOLD_FIRST_DIR / "valid_molecules.csv"
print("Using scaffold-first candidates for Cell 19:")
print(VALID_CSV_FOR_TASK_FILTER)


# %%
# Cell 19: apply strict chemistry/property filters and the external predictor.
# Set ROW_IDX and either TARGET_Y or MIN_PRED_Y according to your Pu_logD task.

ROW_IDX = 0
TARGET_Y = None
MIN_PRED_Y = 0.6
TARGET_WINDOW = 0.7

cmd = [
    sys.executable,
    str(PROJECT_ROOT / "scripts" / "server" / "unigem_task_filter_with_predictor.py"),
    "--valid_csv",
    str(globals().get("VALID_CSV_FOR_TASK_FILTER", TASK_MODEL_PATH / "eval" / "valid_molecules" / "valid_molecules.csv")),
    "--output_dir",
    str(TASK_MODEL_PATH / "eval" / "task_filtered"),
    "--row_idx",
    str(ROW_IDX),
    "--target_window",
    str(TARGET_WINDOW),
    "--top_k",
    "200",
    "--reward_buffer_csv",
    str(TASK_MODEL_PATH / "rl_reward_buffer.csv"),
    "--relaxed_protic_filter",
    "--max_hbd",
    "1",
    "--max_mw",
    "700",
    "--max_rot_bonds",
    "14",
    "--min_qed",
    "0.25",
    "--max_largest_ring_size",
    "12",
    "--max_synth_proxy",
    "9.0",
    "--min_aromatic_ring_count",
    "2",
    "--max_phosphorus_atoms",
    "0",
    "--exclude_phosphorus",
    "--require_amide",
    "--require_aromatic_n",
    "--require_aromatic_amide_like",
    "--min_pred_rf",
    "0.1",
    "--max_model_disagreement",
    "1.0",
    "--predictor_weight",
    "0.85",
    "--property_weight",
    "0.15",
    "--sort_by",
    "pred_y",
    "--require_task_scaffold_or_motif",
]
if TARGET_Y is not None:
    cmd += ["--target_y", str(TARGET_Y)]
if MIN_PRED_Y is not None:
    cmd += ["--min_pred_y", str(MIN_PRED_Y)]

subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


# %%
# Cell 20: optional closed-loop reward cycles.
# This repeats generation and scoring, then appends top reward molecules to rl_reward_buffer.csv.

ROW_IDX = 0
TARGET_Y = None
MIN_PRED_Y = None

cmd = [
    sys.executable,
    str(PROJECT_ROOT / "scripts" / "server" / "unigem_closed_loop_driver.py"),
    "--model_path",
    str(TASK_MODEL_PATH),
    "--checkpoint_epoch",
    "final",
    "--cycles",
    "3",
    "--n_samples",
    "1000",
    "--batch_size_gen",
    "100",
    "--sample_steps",
    "1000",
    "--row_idx",
    str(ROW_IDX),
    "--target_window",
    "0.7",
    "--top_k",
    "200",
    "--append_reward_buffer",
    "--relaxed_protic_filter",
    "--max_hbd",
    "4",
    "--max_mw",
    "700",
    "--require_task_scaffold_or_motif",
]
if TARGET_Y is not None:
    cmd += ["--target_y", str(TARGET_Y)]
if MIN_PRED_Y is not None:
    cmd += ["--min_pred_y", str(MIN_PRED_Y)]

subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


# %%
# Cell 21: create final task figures and paper-ready tables for the latest single-run task filter.

subprocess.run(
    [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "server" / "make_task_report_figures.py"),
        "--task_dir",
        str(TASK_MODEL_PATH / "eval" / "task_filtered"),
        "--reward_buffer_csv",
        str(TASK_MODEL_PATH / "rl_reward_buffer.csv"),
        "--output_dir",
        str(TASK_MODEL_PATH / "eval" / "task_filtered" / "report_figures"),
    ],
    cwd=PROJECT_ROOT,
    check=True,
)

print("Final task report figures:")
print(TASK_MODEL_PATH / "eval" / "task_filtered" / "report_figures")


# %%
# Cell 22: predictor-guided generation loop.
# This is the real feedback loop: generate -> predictor score -> high-reward dataset -> finetune next generator.
# Run this after Cell 17.5 if you want the predictor to guide later generations, not just screen them.

GUIDED_START_MODEL_PATH = TASK_FINETUNE_MODEL_PATH

cmd = [
    sys.executable,
    str(PROJECT_ROOT / "scripts" / "server" / "unigem_predictor_guided_generation.py"),
    "--start_model_path",
    str(GUIDED_START_MODEL_PATH),
    "--guided_exp_prefix",
    "task_qm9_unigem_predictor_guided",
    "--cycles",
    "3",
    "--n_samples",
    "2000",
    "--batch_size_gen",
    "100",
    "--sample_steps",
    "1000",
    "--row_idx",
    "0",
    "--target_window",
    "0.7",
    "--top_k",
    "300",
    "--finetune_epochs",
    "80",
    "--finetune_batch_size",
    "16",
    "--dataset",
    "qm9_task",
    "--relaxed_protic_filter",
    "--max_hbd",
    "4",
    "--max_mw",
    "700",
    "--require_task_scaffold_or_motif",
]

# Optional task controls:
# cmd += ["--target_y", "YOUR_TARGET_VALUE"]
# cmd += ["--min_pred_y", "YOUR_MIN_PRED_VALUE"]

subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
