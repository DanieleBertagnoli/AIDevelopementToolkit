import logging
import os
import shutil
import tempfile

import numpy as np
from torchvision.models import resnet18

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger
from aidevelopementtoolkit.torch_utils.model_utils import *

if __name__ == "__main__":

    logger = get_formatted_logger()

    logger.info("Starting model_utils integration test.")

    # Model configuration
    model_config = {
        "weights": None,      # Don't download pretrained weights
        "num_classes": 10,
    }

    model = resnet18(**model_config)

    checkpoint_dir = tempfile.mkdtemp(prefix="test_checkpoint_")
    logger.info("Temporary checkpoint directory: %s", checkpoint_dir)

    try:
        # Test save_model()
        logger.info("Testing save_model()")

        save_model(model, model_config, checkpoint_dir)

        assert os.path.exists(os.path.join(checkpoint_dir, "weights.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "model_configs.json"))

        logger.info("✓ save_model() passed")

        # Test load_model()
        logger.info("Testing load_model()")

        loaded_model = load_model(resnet18, checkpoint_dir)
        loaded_model.eval()

        logger.info("✓ load_model() passed")

        # Test export_to_onnx()
        logger.info("Testing export_to_onnx()")

        onnx_path = os.path.join(checkpoint_dir, "resnet18.onnx")

        export_to_onnx(
            model=loaded_model,
            input_shape=[(3, 224, 224)],
            input_names=["image"],
            output_names=["logits"],
            export_path=onnx_path,
        )

        assert os.path.exists(onnx_path)

        logger.info("✓ export_to_onnx() passed")

        # Test run_onnx()
        logger.info("Testing run_onnx()")

        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        outputs = run_onnx(
            onnx_path,
            {"image": dummy_input},
        )

        assert len(outputs) == 1
        assert outputs[0].shape == (1, model_config["num_classes"])

        logger.info(
            "✓ run_onnx() passed (output shape: %s)",
            outputs[0].shape,
        )

        # Test print_model_summary()
        logger.info("Testing print_model_summary()")

        print_model_summary(
            loaded_model,
            (3, 224, 224),
        )

        logger.info("✓ print_model_summary() passed")

        logger.info("All integration tests passed successfully.")

    finally:
        shutil.rmtree(checkpoint_dir)
        logger.info("Temporary checkpoint directory removed.")