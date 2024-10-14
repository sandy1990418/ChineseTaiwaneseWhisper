import mlflow
import os
import json
from abc import ABC, abstractmethod
from peft import PeftModel
from src.utils.logging import logger


def setup_mlflow():
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(checkpoint_dir, latest_checkpoint)


class MLflowLogger(ABC):
    @abstractmethod
    def log_model(
        self,
        lora_model: PeftModel,
        checkpoint_dir: str,
        base_model_name: str,
        model_name: str,
    ):
        pass


class WhisperLoRAMLflowLogger(MLflowLogger):
    def log_model(
        self,
        lora_model: PeftModel,
        checkpoint_dir: str,
        base_model_name: str,
        model_name: str,
    ):
        with mlflow.start_run():
            # Log base model name
            mlflow.log_param("base_model_name", base_model_name)

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

            # Log LoRA files
            artifact_path = "latest_lora_checkpoint"
            lora_files = [
                "adapter_config.json",
                "adapter_model.bin",
                "README.md",
                "training_args.bin",
            ]

            for file in lora_files:
                file_path = os.path.join(latest_checkpoint, file)
                if os.path.exists(file_path):
                    mlflow.log_artifact(file_path, artifact_path)

            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            registered_model = mlflow.register_model(model_uri, model_name)

            logger.info(
                f"Latest LoRA checkpoint registered with name: {model_name}, version: {registered_model.version}"
            )


class MLflowLoggerFactory:
    @staticmethod
    def get_logger(model_type: str) -> MLflowLogger:
        if model_type == "whisper_lora":
            return WhisperLoRAMLflowLogger()
        # Add more loggers for different model types if needed
        raise ValueError(f"Unsupported model type: {model_type}")


def mlflow_logging(experiment_name: str, model_type: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if os.getenv("USE_MLFLOW", "false").lower() == "true":
                setup_mlflow()
                mlflow.set_experiment(experiment_name)
                result = func(*args, **kwargs)

                if (
                    isinstance(result, dict)
                    and "lora_model" in result
                    and "checkpoint_dir" in result
                    and "base_model_name" in result
                ):
                    logger = MLflowLoggerFactory.get_logger(model_type)
                    lora_model = result["lora_model"]
                    checkpoint_dir = result["checkpoint_dir"]
                    base_model_name = result["base_model_name"]
                    model_name = f"{experiment_name}_{model_type}_model"

                    logger.log_model(
                        lora_model, checkpoint_dir, base_model_name, model_name
                    )

                    # Log other metrics
                    if "train_metrics" in result:
                        mlflow.log_metrics(result["train_metrics"])
                    if "eval_metrics" in result:
                        mlflow.log_metrics(result["eval_metrics"])

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
