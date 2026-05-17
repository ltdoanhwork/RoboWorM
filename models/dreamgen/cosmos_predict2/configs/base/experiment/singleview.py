# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Post-training experiment configs for three single-view robot datasets:

  - Bimanual ALOHA singleview   (datasets/bimanual_dreamgen, 480x640)
  - Single-arm DROID singleview (datasets/single_arm_dreamgen, 480x640)
  - Humanoid GR1 singleview     (datasets/humanoid_dreamgen, 480x640)

Data preparation (from project root):
    python finetuning/dreamgen/scripts/prepare_bimanual_dreamgen.py
    python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen.py
    python finetuning/dreamgen/scripts/prepare_humanoid_dreamgen.py
    bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh

Training (from project root via finetuning/dreamgen/launch.sh):
    NPROC=8 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_bimanual_singleview
    NPROC=8 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_single_arm_singleview
    NPROC=8 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_humanoid_singleview

Checkpoints saved to:
    checkpoints/posttraining/singleview/{bimanual,single_arm,humanoid}_singleview/
"""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L

# Resolve dataset root absolutely so it works regardless of CWD.
# singleview.py lives at: <project>/models/dreamgen/cosmos_predict2/configs/base/experiment/
# Project root is 6 levels up from that directory.
_SINGLEVIEW_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SINGLEVIEW_DIR, *[".."] * 6))
_DATASETS_ROOT = os.path.join(_PROJECT_ROOT, "datasets")

cs = ConfigStore.instance()


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Shared training hyper-parameters
# ---------------------------------------------------------------------------

_OPTIMIZER = dict(lr=2 ** (-14.5))

_SCHEDULER = dict(
    f_max=[0.2],
    f_min=[0.1],
    warm_up_steps=[1_000],
    cycle_lengths=[10_000],
)

_TRAINER = dict(
    distributed_parallelism="fsdp",
    callbacks=dict(
        iter_speed=dict(hit_thres=200),
    ),
)

_CHECKPOINT = dict(save_iter=1000)

_MODEL = dict(
    config=dict(
        fsdp_shard_size=-1,
        pipe_config=dict(
            guardrail_config=dict(enabled=False),
            prompt_refiner_config=dict(enabled=False),
        ),
    )
)

_MODEL_PARALLEL = dict(context_parallel_size=2)

# ---------------------------------------------------------------------------
# Bimanual ALOHA singleview  (480x640)
# ---------------------------------------------------------------------------

_dataset_bimanual = L(Dataset)(
    dataset_dir=os.path.join(_DATASETS_ROOT, "bimanual_dreamgen"),
    num_frames=93,
    video_size=(432, 768),
)

_dataloader_bimanual = L(DataLoader)(
    dataset=_dataset_bimanual,
    sampler=L(get_sampler)(dataset=_dataset_bimanual),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

cs.store(
    group="dataloader_train",
    package="dataloader_train",
    name="bimanual_singleview",
    node=_dataloader_bimanual,
)

predict2_video2world_training_2b_bimanual_singleview = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_16fps"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        {"override /dataloader_train": "bimanual_singleview"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=_MODEL,
    optimizer=_OPTIMIZER,
    scheduler=_SCHEDULER,
    job=dict(
        project="posttraining",
        group="singleview",
        name="bimanual_singleview",
    ),
    model_parallel=_MODEL_PARALLEL,
    trainer=_TRAINER,
    checkpoint=_CHECKPOINT,
)

# ---------------------------------------------------------------------------
# Single-arm DROID singleview  (480x640)
# ---------------------------------------------------------------------------

_dataset_single_arm = L(Dataset)(
    dataset_dir=os.path.join(_DATASETS_ROOT, "single_arm_dreamgen"),
    num_frames=93,
    video_size=(432, 768),
)

_dataloader_single_arm = L(DataLoader)(
    dataset=_dataset_single_arm,
    sampler=L(get_sampler)(dataset=_dataset_single_arm),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

cs.store(
    group="dataloader_train",
    package="dataloader_train",
    name="single_arm_singleview",
    node=_dataloader_single_arm,
)

predict2_video2world_training_2b_single_arm_singleview = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_16fps"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        {"override /dataloader_train": "single_arm_singleview"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=_MODEL,
    optimizer=_OPTIMIZER,
    scheduler=_SCHEDULER,
    job=dict(
        project="posttraining",
        group="singleview",
        name="single_arm_singleview",
    ),
    model_parallel=_MODEL_PARALLEL,
    trainer=_TRAINER,
    checkpoint=_CHECKPOINT,
)

# ---------------------------------------------------------------------------
# Humanoid GR1 singleview  (480x640)
# ---------------------------------------------------------------------------

_dataset_humanoid = L(Dataset)(
    dataset_dir=os.path.join(_DATASETS_ROOT, "humanoid_dreamgen"),
    num_frames=93,
    video_size=(432, 768),
)

_dataloader_humanoid = L(DataLoader)(
    dataset=_dataset_humanoid,
    sampler=L(get_sampler)(dataset=_dataset_humanoid),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

cs.store(
    group="dataloader_train",
    package="dataloader_train",
    name="humanoid_singleview",
    node=_dataloader_humanoid,
)

predict2_video2world_training_2b_humanoid_singleview = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_16fps"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        {"override /dataloader_train": "humanoid_singleview"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=_MODEL,
    optimizer=_OPTIMIZER,
    scheduler=_SCHEDULER,
    job=dict(
        project="posttraining",
        group="singleview",
        name="humanoid_singleview",
    ),
    model_parallel=_MODEL_PARALLEL,
    trainer=_TRAINER,
    checkpoint=_CHECKPOINT,
)

# ---------------------------------------------------------------------------
# All singleview combined  (bimanual + single_arm + humanoid, 480x640)
# Uses ConcatDataset so the model sees all three embodiments in one run.
# ---------------------------------------------------------------------------

_dataset_all = L(ConcatDataset)(
    datasets=[_dataset_bimanual, _dataset_single_arm, _dataset_humanoid]
)

_dataloader_all = L(DataLoader)(
    dataset=_dataset_all,
    sampler=L(get_sampler)(dataset=_dataset_all),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

cs.store(
    group="dataloader_train",
    package="dataloader_train",
    name="all_singleview",
    node=_dataloader_all,
)

predict2_video2world_training_2b_all_singleview = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_16fps"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        {"override /dataloader_train": "all_singleview"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=_MODEL,
    optimizer=_OPTIMIZER,
    scheduler=_SCHEDULER,
    job=dict(
        project="posttraining",
        group="singleview",
        name="all_singleview",
    ),
    model_parallel=_MODEL_PARALLEL,
    trainer=_TRAINER,
    checkpoint=_CHECKPOINT,
)

# ---------------------------------------------------------------------------
# Humanoid multiview  (4 cameras × episodes, 480x640)
# ---------------------------------------------------------------------------

_dataset_humanoid_multiview = L(Dataset)(
    dataset_dir=os.path.join(_DATASETS_ROOT, "humanoid_multiview_dreamgen"),
    num_frames=93,
    video_size=(432, 768),
)

_dataloader_humanoid_multiview = L(DataLoader)(
    dataset=_dataset_humanoid_multiview,
    sampler=L(get_sampler)(dataset=_dataset_humanoid_multiview),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

cs.store(
    group="dataloader_train",
    package="dataloader_train",
    name="humanoid_multiview",
    node=_dataloader_humanoid_multiview,
)

predict2_video2world_training_2b_humanoid_multiview = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_16fps"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        {"override /dataloader_train": "humanoid_multiview"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=_MODEL,
    optimizer=_OPTIMIZER,
    scheduler=_SCHEDULER,
    job=dict(
        project="posttraining",
        group="multiview",
        name="humanoid_multiview",
    ),
    model_parallel=_MODEL_PARALLEL,
    trainer=_TRAINER,
    checkpoint=_CHECKPOINT,
)

# ---------------------------------------------------------------------------
# Bimanual multiview  (4 cameras × episodes, 480x640)
# ---------------------------------------------------------------------------

_dataset_bimanual_multiview = L(Dataset)(
    dataset_dir=os.path.join(_DATASETS_ROOT, "bimanual_multiview_dreamgen"),
    num_frames=93,
    video_size=(432, 768),
)

_dataloader_bimanual_multiview = L(DataLoader)(
    dataset=_dataset_bimanual_multiview,
    sampler=L(get_sampler)(dataset=_dataset_bimanual_multiview),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

cs.store(
    group="dataloader_train",
    package="dataloader_train",
    name="bimanual_multiview",
    node=_dataloader_bimanual_multiview,
)

predict2_video2world_training_2b_bimanual_multiview = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_16fps"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        {"override /dataloader_train": "bimanual_multiview"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=_MODEL,
    optimizer=_OPTIMIZER,
    scheduler=_SCHEDULER,
    job=dict(
        project="posttraining",
        group="multiview",
        name="bimanual_multiview",
    ),
    model_parallel=_MODEL_PARALLEL,
    trainer=_TRAINER,
    checkpoint=_CHECKPOINT,
)

# ---------------------------------------------------------------------------
# All multiview combined  (humanoid + bimanual, 480x640)
# ---------------------------------------------------------------------------

_dataset_all_multiview = L(ConcatDataset)(
    datasets=[_dataset_humanoid_multiview, _dataset_bimanual_multiview]
)

_dataloader_all_multiview = L(DataLoader)(
    dataset=_dataset_all_multiview,
    sampler=L(get_sampler)(dataset=_dataset_all_multiview),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

cs.store(
    group="dataloader_train",
    package="dataloader_train",
    name="all_multiview",
    node=_dataloader_all_multiview,
)

predict2_video2world_training_2b_all_multiview = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_16fps"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        {"override /dataloader_train": "all_multiview"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=_MODEL,
    optimizer=_OPTIMIZER,
    scheduler=_SCHEDULER,
    job=dict(
        project="posttraining",
        group="multiview",
        name="all_multiview",
    ),
    model_parallel=_MODEL_PARALLEL,
    trainer=_TRAINER,
    checkpoint=_CHECKPOINT,
)

# ---------------------------------------------------------------------------
# Register all experiments with ConfigStore
# ---------------------------------------------------------------------------

for _item in [
    predict2_video2world_training_2b_bimanual_singleview,
    predict2_video2world_training_2b_single_arm_singleview,
    predict2_video2world_training_2b_humanoid_singleview,
    predict2_video2world_training_2b_all_singleview,
    predict2_video2world_training_2b_humanoid_multiview,
    predict2_video2world_training_2b_bimanual_multiview,
    predict2_video2world_training_2b_all_multiview,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
