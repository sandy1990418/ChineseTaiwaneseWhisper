import time
import random
from src.utils.mlflow_logging import mlflow_logging, setup_mlflow
import torch
import torch.nn as nn
from src.utils.logging import logger
from dotenv import load_dotenv

load_dotenv()

# # Set environment variables
# os.environ["USE_MLFLOW"] = "true"
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"  # Adjust as needed


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@mlflow_logging(experiment_name="Decorator_Test")
def train_model(epochs, learning_rate):
    logger.info(
        f"Starting model training with {epochs} epochs and learning rate {learning_rate}"
    )
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Simulate training
        inputs = torch.randn(100, 10)
        targets = torch.randn(100, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        time.sleep(0.1)  # Simulate some processing time

    # Simulate evaluation
    eval_loss = random.uniform(0.1, 0.5)
    accuracy = random.uniform(0.7, 0.95)

    logger.info(
        f"Training completed. Eval Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}"
    )

    return {
        "model": model,  # 返回模型的狀態字典，而不是模型對象
        "train_metrics": {"final_loss": loss.item()},
        "eval_metrics": {"eval_loss": eval_loss, "accuracy": accuracy},
    }


if __name__ == "__main__":
    logger.info("Starting MLflow decorator test")
    try:
        setup_mlflow()
        result = train_model(epochs=5, learning_rate=0.01)
        logger.info("Training completed. Check MLflow for logged data.")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}")
        logger.error("Make sure the MLflow server is running and accessible.")
