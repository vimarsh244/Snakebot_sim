"""WandB utilities."""

from __future__ import annotations

import os
import shutil
from typing import Sequence


def patch_wandb_save_file_for_windows() -> None:
  """Patch rsl_rl's WandbSummaryWriter.save_file so file upload works on Windows.

  On Windows, wandb.save(..., base_path=...) creates symlinks, which require
  admin/developer mode (OSError 1314). We patch save_file to catch that and copy
  the file into the wandb run dir instead, then call wandb.save on the copy.
  """
  try:
    import wandb
    from rsl_rl.utils import wandb_utils
  except ImportError:
    return

  def _wandb_save_copy_fallback(path: str, base_path: str | None = None) -> None:
    """Call wandb.save; on Windows symlink error, copy file into run dir instead."""
    try:
      wandb.save(path, base_path=base_path or os.path.dirname(path))
    except OSError as e:
      if getattr(e, "winerror", None) != 1314 and "symlink" not in str(e).lower():
        raise
      if not wandb.run or not os.path.isfile(path):
        raise
      run_dir = wandb.run.dir
      name = os.path.basename(path)
      dest = os.path.join(run_dir, "files", name)
      os.makedirs(os.path.dirname(dest), exist_ok=True)
      shutil.copy2(path, dest)

  _original_save_file = wandb_utils.WandbSummaryWriter.save_file

  def save_file_copy_fallback(self: wandb_utils.WandbSummaryWriter, path: str) -> None:
    _wandb_save_copy_fallback(path, base_path=os.path.dirname(path))

  _original_save_model = wandb_utils.WandbSummaryWriter.save_model

  def save_model_copy_fallback(
    self: wandb_utils.WandbSummaryWriter, model_path: str, it: int
  ) -> None:
    _wandb_save_copy_fallback(model_path, base_path=os.path.dirname(model_path))

  wandb_utils.WandbSummaryWriter.save_file = save_file_copy_fallback
  wandb_utils.WandbSummaryWriter.save_model = save_model_copy_fallback


def add_wandb_tags(tags: Sequence[str]) -> None:
  """Add tags to the current wandb run.

  Note: This function stores tags in wandb.config._wandb_tags if the run is not yet
  initialized, allowing them to be retrieved later. If the run is already initialized,
  tags are added directly.
  """
  if not tags:
    return

  try:
    import wandb

    if wandb.run is not None:
      existing_tags = list(wandb.run.tags) if wandb.run.tags else []
      new_tags = list(set(existing_tags + list(tags)))
      wandb.run.tags = new_tags
    else:
      # Store tags to be added when run is initialized.
      # This is a workaround for lazy wandb initialization in rsl_rl 3.1.0.
      current_tags = os.environ.get("WANDB_TAGS", "")
      all_tags = set(current_tags.split(",") if current_tags else [])
      all_tags.update(tags)
      os.environ["WANDB_TAGS"] = ",".join(sorted(all_tags))
  except ImportError:
    pass
