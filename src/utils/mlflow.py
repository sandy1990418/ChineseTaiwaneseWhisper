import functools
import mlflow
import os
import time
from mlflow.models import infer_signature


def mlflow_logging(experiment_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.getenv("USE_MLFLOW", "false").lower() == "true":
                mlflow.set_tracking_uri(
                    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
                )
                mlflow.set_experiment(experiment_name)

                with mlflow.start_run():
                    # record parameters
                    mlflow.log_params(kwargs)

                    # record start time
                    start_time = time.time()

                    # operate original function
                    result = func(*args, **kwargs)

                    # record start and end time
                    end_time = time.time()
                    duration = end_time - start_time
                    mlflow.log_metric("duration", duration)

                    # record result
                    if isinstance(result, dict):
                        mlflow.log_metrics(result)

                    # register model
                    if "model" in result and "eval_metrics" in result:
                        model = result["model"]
                        eval_metrics = result["eval_metrics"]

                        # generate model sign 
                        signature = infer_signature(args[0], eval_metrics)

                        # register model 
                        mlflow.pytorch.log_model(
                            model,
                            "model",
                            signature=signature,
                            registered_model_name=f"{experiment_name}_model",
                        )

                    return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
