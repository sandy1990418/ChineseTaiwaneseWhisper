import mlflow
import os
import json
from abc import ABC, abstractmethod
from peft import PeftModel
from src.utils.logging import logger
import time
from time import strftime,  localtime
from typing import Dict
import functools
from dotenv import load_dotenv


load_dotenv(dotenv_path='./.env')

LORA_LIST = ['lora', 'qlora', 'olora']


def setup_mlflow():
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")


def get_latest_checkpoint(base_path: str) -> str:
    if os.path.exists(os.path.join(base_path, "adapter_config.json")):
        return base_path
    checkpoints = [d for d in os.listdir(base_path) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(base_path, latest_checkpoint)


def extract_metrics_from_trainer_state(trainer_state_path: str) -> Dict[str, float]:
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    if 'log_history' in trainer_state and trainer_state['log_history']:
        last_log = trainer_state['log_history']
        metrics = {k: v for k, v in last_log.items() if isinstance(v, (int, float))}
        return metrics
    
    return {}


class MLflowLogger(ABC):
    @abstractmethod
    def log_model(
        self,
        lora_model: PeftModel,
        checkpoint_dir: str,
        base_model_name: str,
        model_name: str,
    ):
        raise NotImplementedError("If you want to customize MLflow to track your work, please implement it here.")


class WhisperLoRAMLflowLogger(MLflowLogger):
    def log_model(
        self,
        experiment_name: str,
        # lora_model: PeftModel,
        checkpoint_dir: str,
        base_model_name: str,
        data_config: dict,
        train_dataset: str
        # model_name: str,
    ):
        start_time = time.time()
        run_name = f"{experiment_name}_{strftime('%Y-%m-%d %H:%M:%S', localtime(start_time))}"

        with mlflow.start_run(run_name=run_name):
            logger.info("Log Base model name")
            # Log base model name
            mlflow.log_param("base_model_name", base_model_name)
            mlflow.log_param("train_dataset", train_dataset)
            for key, value in data_config.items():
                mlflow.log_param(key, value)
            # Get the latest checkpoint
            latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
            if not latest_checkpoint:
                logger.warning("No checkpoint found. Logging skipped.")
                return

            # Log LoRA configuration
            config_path = os.path.join(latest_checkpoint, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    mlflow.log_params(config)

            # Extract and log metrics from trainer_state.json
            trainer_state_path = os.path.join(latest_checkpoint, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                metrics = extract_metrics_from_trainer_state(trainer_state_path)
                mlflow.log_metrics(metrics)

            # Log LoRA files
            artifact_path = "lora_files"
            lora_files = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "added_tokens.json",
                "all_results.json",
                "merges.txt",
                "normalizer.json",
                "preprocessor_config.json",
                "README.md",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "train_results.json",
                "trainer_state.json",
                "training_args.bin",
                "vocab.json"
            ]

            for file in lora_files:
                file_path = os.path.join(latest_checkpoint, file)
                if os.path.exists(file_path):
                    mlflow.log_artifact(file_path, artifact_path)

        mlflow.end_run(status='FINISHED')
        # Register model
        # model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        # registered_model = mlflow.register_model(model_uri, model_name)

        # logger.info(
        #     f"Latest LoRA checkpoint registered with name: {model_name}, version: {registered_model.version}"
        # )


class MLflowLoggerFactory:
    @staticmethod
    def get_logger(finetune_type: str) -> MLflowLogger:
        if finetune_type in LORA_LIST:
            return WhisperLoRAMLflowLogger().log_model
        # Add more loggers for different model types if needed
        raise ValueError(f"Unsupported model type: {finetune_type}")


def mlflow_logging(experiment_name: str, model_type: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.getenv("USE_MLFLOW", "false").lower() == "true":
                result = func(*args, **kwargs)
                setup_mlflow()
                mlflow.set_experiment(experiment_name)
                mllogger = MLflowLoggerFactory.get_logger(model_type)
                logger.info("Finish Training")
                if (
                    isinstance(result, dict)
                    and "checkpoint_dir" in result
                    and "base_model_name" in result
                    and "data_config" in result
                    and "train_dataset" in result
                ):  
                    checkpoint_dir = result["checkpoint_dir"]
                    base_model_name = result["base_model_name"]
                    data_config = result['data_config']
                    train_dataset = result['train_dataset']
                    # model_name = f"{experiment_name}_{model_type}_model"

                    mllogger(
                        experiment_name=experiment_name,
                        checkpoint_dir=checkpoint_dir, 
                        base_model_name=base_model_name,
                        data_config=data_config,
                        train_dataset=train_dataset
                    )
                    return result
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator


# def mlflow_logging(experiment_name):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             if os.getenv("USE_MLFLOW", "false").lower() == "true":
#                 setup_mlflow()
#                 try:
#                     mlflow.set_experiment(experiment_name)
#                     # record start time
#                     start_time = time.time()
#                     run_name = f"{experiment_name} {strftime('%Y-%m-%d %H:%M:%S', localtime(start_time))}"
#                     with mlflow.start_run(run_name=run_name):
#                         logger.info(
#                             f"Starting MLflow run for experiment: {experiment_name}"
#                         )

#                         # Execute the original function
#                         result = func(*args, **kwargs)

#                         # Log parameters
#                         mlflow.log_params(kwargs)

#                         # record start and end time
#                         end_time = time.time()
#                         duration = end_time - start_time
#                         mlflow.log_metric("duration", duration)

#                         # record result
#                         if isinstance(result, dict):
#                             if "train_metrics" in result:
#                                 mlflow.log_metrics(result["train_metrics"])
#                             if "eval_metrics" in result:
#                                 mlflow.log_metrics(result["eval_metrics"])

#                         # Log model
#                         if 'model' in result and isinstance(result['model'], torch.nn.Module):
#                             mlflow.pytorch.log_model(result["model"], "model")
#                             # Register the model
#                             model_version = mlflow.register_model(
#                                 f"runs:/{mlflow.active_run().info.run_id}/model",
#                                 f"{experiment_name}_model"
#                             )
#                             logger.info(f"Model registered with version: {model_version.version}")
#                         logger.info(
#                             f"MLflow run completed for experiment: {experiment_name}"
#                         )
#                         return result
#                 except mlflow.exceptions.MlflowException as e:
#                     logger.error(f"Failed to set up MLflow experiment: {e}")
#                     logger.warning("Continuing without MLflow logging")
#                     return func(*args, **kwargs)
#             else:
#                 logger.info(
#                     "MLflow logging is disabled. Running function without MLflow tracking."
#                 )
#                 return func(*args, **kwargs)

#         return wrapper

#     return decorator
