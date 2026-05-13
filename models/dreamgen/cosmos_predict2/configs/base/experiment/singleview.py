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

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L

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

_CHECKPOINT = dict(save_iter=500)

_MODEL = dict(
    config=dict(
        fsdp_shard_size=8,
        pipe_config=dict(
            state_t=93,
            guardrail_config=dict(enabled=False),
            prompt_refiner_config=dict(enabled=False),
        ),
    )
)

_MODEL_PARALLEL = dict(context_parallel_size=1)

# ---------------------------------------------------------------------------
# Bimanual ALOHA singleview  (480x640)
# ---------------------------------------------------------------------------

_dataset_bimanual = L(Dataset)(
    dataset_dir="datasets/bimanual_dreamgen",
    num_frames=93,
    video_size=(480, 640),
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
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
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
    dataset_dir="datasets/single_arm_dreamgen",
    num_frames=93,
    video_size=(480, 640),
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
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
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
    dataset_dir="datasets/humanoid_dreamgen",
    num_frames=93,
    video_size=(480, 640),
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
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
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
# Register all experiments with ConfigStore
# ---------------------------------------------------------------------------

for _item in [
    predict2_video2world_training_2b_bimanual_singleview,
    predict2_video2world_training_2b_single_arm_singleview,
    predict2_video2world_training_2b_humanoid_singleview,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
