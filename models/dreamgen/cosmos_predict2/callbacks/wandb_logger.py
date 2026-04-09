# File: cosmos_predict2/callbacks/wandb_logger.py

import wandb
import torch
from imaginaire.utils.callback import Callback
# from imaginaire.utils.distributed import is_master
from imaginaire.config import Config
import torch.distributed as dist

# def is_master_process():
#     """Checks if the current process is the master process (rank 0)."""
#     if not dist.is_available() or not dist.is_initialized():
#         return True
#     return dist.get_rank() == 0

# class WandbLoggerCallback(Callback):
#     """
#     A callback that logs metrics and config to Weights & Biases.
#     """
#     def __init__(self, project, entity=None, name=None, group=None, log_every_n_steps=1):
#         self.project = project
#         self.entity = entity
#         self.name = name
#         self.group = group
#         self.log_every_n_steps = log_every_n_steps
#         self._wandb_run = None

#     # --- CORRECTION 1: Update the function signature ---
#     # It receives the `model` object, not `trainer` and `config`.
#     def on_train_start(self, model, **kwargs):
#         """Called at the beginning of training."""
#         if is_master_process():
#             print("🚀 Initializing Weights & Biases...")
            
#             # --- CORRECTION 2: Access config via self.trainer ---
#             config = self.trainer.config
#             config_dict = Config.to_dict(config)
            
#             self._wandb_run = wandb.init(
#                 project=self.project,
#                 entity=self.entity,
#                 name=self.name or config.job.name,
#                 group=self.group or config.job.group,
#                 config=config_dict,
#                 reinit=True,
#             )
#             print("✅ Weights & Biases initialized successfully.")

#     # --- CORRECTION 3: Proactively update other methods for consistency ---
#     def on_train_batch_end(self, **kwargs):
#         """Called at the end of a training batch."""
#         if is_master_process() and self._wandb_run:
#             # Use self.trainer here as well.
#             current_step = self.trainer.state_dict['step']
#             if current_step % self.log_every_n_steps == 0:
#                 metrics_to_log = {}
#                 if 'metric_storage' in self.trainer.state_dict:
#                     for key, value in self.trainer.state_dict['metric_storage'].items():
#                         if value.latest is not None:
#                              metrics_to_log[key] = value.latest
                
#                 if metrics_to_log:
#                     self._wandb_run.log(metrics_to_log, step=current_step)

#     def on_train_end(self, **kwargs):
#         """Called at the end of training."""
#         if is_master_process() and self._wandb_run:
#             print("Finishing Weights & Biases run...")
#             self._wandb_run.finish()

def is_master_process():
    """Checks if the current process is the master process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


### Old Loging without Loss Value 

# class WandbLoggerCallback(Callback):
#     """A callback that logs metrics and config to Weights & Biases."""
    
#     def __init__(self, project, entity=None, name=None, group=None, log_every_n_steps=1):
#         self.project = project
#         self.entity = entity
#         self.name = name
#         self.group = group
#         self.log_every_n_steps = log_every_n_steps
#         self._wandb_run = None

#     def on_train_start(self, model, **kwargs):
#         """Called at the beginning of training."""
#         if is_master_process():
#             print("🚀 Initializing Weights & Biases...")
            
#             # --- THIS IS THE NEW, SIMPLIFIED APPROACH ---
#             # Instead of converting the complex config object, we create a
#             # simple dictionary with the most important parameters.
            
#             # Get the main config object from the trainer.
#             trainer_config = self.trainer.config
            
#             # Build a simple, safe dictionary for wandb.
#             simple_config = {
#                 # "experiment": trainer_config.experiment,
#                 "learning_rate": trainer_config.optimizer.lr,
#                 "batch_size": trainer_config.dataloader_train.batch_size,
#                 # "max_iter": trainer_config.max_iter,
#                 "job_name": trainer_config.job.name,
#                 "job_group": trainer_config.job.group,
#             }

#             self._wandb_run = wandb.init(
#                 project=self.project,
#                 entity=self.entity,
#                 name=self.name or trainer_config.job.name,
#                 group=self.group or trainer_config.job.group,
#                 config=simple_config,  # Use the new, simple config
#             )
#             print("✅ Weights & Biases initialized successfully.")

#     def on_train_batch_end(self, **kwargs):
#         """Called at the end of a training batch."""
#         if is_master_process() and self._wandb_run:
#             current_step = self.trainer.state_dict['step']
#             if current_step % self.log_every_n_steps == 0:
#                 metrics_to_log = {}
#                 if 'metric_storage' in self.trainer.state_dict:
#                     for key, value in self.trainer.state_dict['metric_storage'].items():
#                         if value.latest is not None:
#                              metrics_to_log[key] = value.latest
                
#                 if metrics_to_log:
#                     self._wandb_run.log(metrics_to_log, step=current_step)

#     def on_train_end(self, **kwargs):
#         """Called at the end of training."""
#         if is_master_process() and self._wandb_run:
#             print("Finishing Weights & Biases run...")
#             self._wandb_run.finish()



### Updating Caoll back with Loss value 
class WandbLoggerCallback(Callback):
    """A callback that logs metrics and config to Weights & Biases."""
    
    def __init__(self, project, entity=None, name=None, group=None, log_every_n_steps=1):
        self.project = project
        self.entity = entity
        self.name = name
        self.group = group
        self.log_every_n_steps = log_every_n_steps
        self._wandb_run = None

    def on_train_start(self, model, **kwargs):
        """Called at the beginning of training."""
        if is_master_process():
            print("🚀 Initializing Weights & Biases...")
            trainer_config = self.trainer.config
            simple_config = {
                # "experiment": trainer_config.job.name,
                "learning_rate": trainer_config.optimizer.lr,
                "batch_size": trainer_config.dataloader_train.batch_size,
                # "max_steps": trainer_config.max_steps,
                "job_name": trainer_config.job.name,
                "job_group": trainer_config.job.group,
            }

            self._wandb_run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name or trainer_config.job.name,
                group=self.group or trainer_config.job.group,
                config=simple_config,
            )
            print("✅ Weights & Biases initialized successfully.")

    # --- THIS METHOD IS NOW CORRECTED ---
    def on_train_batch_end(self, **kwargs):
        """Called at the end of a training batch."""
        if is_master_process() and self._wandb_run:
            current_step = self.trainer.state_dict['step']
            if current_step % self.log_every_n_steps == 0:
                metrics_to_log = {}
                
                # First, get all generic metrics from metric_storage
                if 'metric_storage' in self.trainer.state_dict:
                    for key, value in self.trainer.state_dict['metric_storage'].items():
                        if value.latest is not None:
                             metrics_to_log[key] = value.latest
                
                # --- ADD THIS SECTION TO EXPLICITLY LOG THE LOSS ---
                # The trainer stores the most recent loss directly in the state_dict.
                if 'loss' in self.trainer.state_dict:
                    # Check if the loss is a tensor and get its item value
                    loss_value = self.trainer.state_dict['loss']
                    if hasattr(loss_value, 'item'):
                        metrics_to_log['loss'] = loss_value.item()
                    else:
                        metrics_to_log['loss'] = loss_value
                # --- END OF NEW SECTION ---

                if metrics_to_log:
                    self._wandb_run.log(metrics_to_log, step=current_step)

    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        if is_master_process() and self._wandb_run:
            print("Finishing Weights & Biases run...")
            self._wandb_run.finish()