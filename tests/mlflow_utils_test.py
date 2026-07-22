import logging
import os
from unittest.mock import patch

import torch.nn as nn
from torchvision.models import resnet18

from aidevelopementtoolkit.logging_utils.mlflow_utils import *
from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

if __name__ == "__main__":
    import logging
    import os
    import shutil
    import tempfile

    import mlflow
    import torch.nn as nn
    from torchvision.models import resnet18

    logger = logging.getLogger(__name__)

    logger.info("Starting mlflow_utils integration test.")

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = resnet18(weights=None, num_classes=10)

        def forward(self, x):
            return self.model(x)

    model = TestModel()

    model_config = {
        "num_classes": 10,
    }

    temp_dir = tempfile.mkdtemp(prefix="mlflow_test_")
    mlflow_db = os.path.join(temp_dir, "mlflow.db")

    os.environ["MLFLOW_ENDPOINT_URL"] = f"sqlite:///{mlflow_db}"

    logger.info(
        "Using temporary MLflow directory: %s",
        temp_dir,
    )

    try:
        logger.info("Testing start_mlflow_run()")

        start_mlflow_run(
            experiment_name="test_experiment",
            mlflow_kwargs={
                "run_name": "integration_test",
            },
        )

        logger.info("✓ start_mlflow_run() passed")

        logger.info("Testing log_run_parameters()")

        parameters = {
            "learning_rate": 1e-3,
            "epochs": 5,
            "optimizer": {
                "name": "Adam",
                "weight_decay": 1e-4,
            },
        }

        log_run_parameters(parameters)

        logger.info("✓ log_run_parameters() passed")

        logger.info("Testing save_model_checkpoint()")

        save_model_checkpoint(
            model=model,
            config=model_config,
            checkpoint_name="best",
        )

        logger.info("✓ save_model_checkpoint() passed")

        logger.info("Testing is_numeric()")

        assert is_numeric(1)
        assert is_numeric(1.5)
        assert not is_numeric("1")
        assert not is_numeric(None)

        logger.info("✓ is_numeric() passed")

        logger.info("Verifying MLflow run data")

        active_run = mlflow.active_run()

        assert active_run is not None

        run_id = active_run.info.run_id

        mlflow.end_run()

        client = mlflow.MlflowClient()

        stored_run = client.get_run(run_id)

        assert stored_run.data.params["learning_rate"] == "0.001"
        assert stored_run.data.params["epochs"] == "5"
        assert stored_run.data.params["optimizer/name"] == "Adam"
        assert stored_run.data.params["optimizer/weight_decay"] == "0.0001"

        artifacts = client.list_artifacts(
            run_id,
            path="model_checkpoints/best",
        )

        artifact_names = [
            artifact.path
            for artifact in artifacts
        ]

        assert any(
            "weights.pt" in artifact
            for artifact in artifact_names
        )

        assert any(
            "model_configs.json" in artifact
            for artifact in artifact_names
        )

        logger.info("✓ MLflow verification passed")

        logger.info("All tests passed successfully.")

        print()
        print("Temporary MLflow data is available at:")
        print(temp_dir)
        print()
        print("You can inspect it with:")
        print(f"  mlflow ui --backend-store-uri sqlite:///{mlflow_db}")
        print()

        input(
            "Press ENTER to remove temporary MLflow data..."
        )

    finally:
        if mlflow.active_run():
            mlflow.end_run()

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

        logger.info(
            "Temporary MLflow directory removed."
        )