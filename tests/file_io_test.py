import logging
import os
import shutil
import tempfile
import zipfile

import h5py
import numpy as np
import pandas as pd

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger
from aidevelopementtoolkit.logging_utils.file_io import save_file, load_file


if __name__ == "__main__":

    logger = get_formatted_logger(level="INFO")
    logger.info("Starting file_utils integration test.")
    test_dir = tempfile.mkdtemp(prefix="test_file_utils_")
    logger.info("Temporary test directory: %s", test_dir,)

    try:

        logger.info("Testing JSON save/load.")
        json_path = os.path.join(test_dir, "data.json")

        json_data = {
            "name": "test",
            "value": 42,
            "active": True,
        }

        save_file(json_data, json_path)

        assert os.path.exists(json_path)
        
        loaded_json = load_file(json_path)
        
        assert loaded_json == json_data
        
        logger.info("✓ JSON save/load passed")
        logger.info("Testing JSON append.")

        json_append_data = {"new_value": 123}

        save_file(json_append_data, json_path, append=True)

        loaded_json = load_file(json_path)

        assert loaded_json["name"] == "test"
        assert loaded_json["value"] == 42
        assert loaded_json["new_value"] == 123

        logger.info("✓ JSON append passed")

        logger.info("Testing YAML save/load.")

        yaml_path = os.path.join(test_dir, "data.yaml")

        yaml_data = {
            "name": "yaml_test",
            "values": [1, 2, 3],
        }

        save_file(yaml_data, yaml_path)
        
        assert os.path.exists(yaml_path)
        
        loaded_yaml = load_file(yaml_path)
        
        assert loaded_yaml == yaml_data
        
        logger.info("✓ YAML save/load passed")

        logger.info("Testing YAML append.")

        save_file({"new_key": "new_value"}, yaml_path, append=True)
        loaded_yaml = load_file(yaml_path)
        
        assert loaded_yaml["name"] == "yaml_test"
        assert loaded_yaml["new_key"] == "new_value"
        
        logger.info("✓ YAML append passed")

        logger.info("Testing CSV save/load as pandas.")

        csv_path = os.path.join(test_dir, "data.csv")

        csv_data = [
            {
                "id": 1,
                "name": "Alice",
                "score": 95.5,
            },
            {
                "id": 2,
                "name": "Bob",
                "score": 87.0,
            },
        ]

        save_file(csv_data, csv_path, header=True)

        assert os.path.exists(csv_path)

        loaded_df = load_file(csv_path, return_type="pandas")

        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == (2, 3)
        assert list(loaded_df.columns) == [
            "id",
            "name",
            "score",
        ]

        assert loaded_df.iloc[0]["name"] == "Alice"
        assert loaded_df.iloc[1]["score"] == 87.0

        logger.info("✓ CSV pandas save/load passed")

        logger.info("Testing CSV load as NumPy.")
        loaded_array = load_file(csv_path, return_type="numpy",)
        
        assert isinstance(loaded_array, np.ndarray)
        assert loaded_array.shape == (2, 3)
        
        logger.info("✓ CSV NumPy load passed")

        logger.info("Testing CSV append.")

        additional_csv_data = [
            {
                "id": 3,
                "name": "Charlie",
                "score": 91.0,
            },
        ]

        save_file(additional_csv_data, csv_path, append=True)
        loaded_df = load_file(csv_path, return_type="pandas")
        
        assert len(loaded_df) == 3
        assert loaded_df.iloc[2]["name"] == "Charlie"

        logger.info("✓ CSV append passed")

        logger.info("Testing NPY save/load.")
        npy_path = os.path.join(test_dir, "data.npy")
        numpy_data = np.random.rand(10, 5).astype(np.float32)
        save_file(numpy_data, npy_path)
        
        assert os.path.exists(npy_path)
        
        loaded_numpy = load_file(npy_path)

        assert isinstance(loaded_numpy, np.ndarray)
        assert loaded_numpy.shape == numpy_data.shape
        assert loaded_numpy.dtype == numpy_data.dtype

        assert np.allclose(loaded_numpy, numpy_data)

        logger.info("✓ NPY save/load passed")

        logger.info("Testing NPZ save/load.")
        npz_path = os.path.join(test_dir, "data.npz")
        first_array = np.array([1, 2, 3])
        second_array = np.array([4, 5])
        save_file({"first": first_array}, npz_path)
        save_file({"second": second_array}, npz_path, append=True)

        loaded_npz = load_file(npz_path)

        assert set(loaded_npz) == {"first", "second"}
        assert np.array_equal(loaded_npz["first"], first_array)
        assert np.array_equal(loaded_npz["second"], second_array)
        with zipfile.ZipFile(npz_path) as npz_archive:
            assert all(
                entry.compress_type == zipfile.ZIP_DEFLATED
                for entry in npz_archive.infolist()
            )

        logger.info("✓ NPZ save/load passed")

        logger.info("Testing H5 save/load.")
        h5_path = os.path.join(test_dir, "data.h5")
        save_file({"first": first_array}, h5_path)
        save_file({"second": second_array}, h5_path, append=True)

        loaded_h5 = load_file(h5_path)
        assert set(loaded_h5) == {"first", "second"}
        assert np.array_equal(loaded_h5["first"], first_array)
        assert np.array_equal(loaded_h5["second"], second_array)
        with h5py.File(h5_path, "r") as h5_archive:
            assert h5_archive["first"].compression == "gzip"
            assert h5_archive["second"].compression == "gzip"

        logger.info("✓ H5 save/load passed")
        
        logger.info("Testing missing file handling.")

        missing_path = os.path.join(test_dir, "missing.json")

        try:
            load_file(missing_path)
            raise AssertionError("Expected FileNotFoundError")

        except FileNotFoundError:
            logger.info("✓ Missing file handling passed")

        logger.info("Testing unsupported extension handling.")

        unsupported_path = os.path.join(test_dir, "data.txt")

        try:
            save_file("some data", unsupported_path,)
            raise AssertionError("Expected ValueError")

        except ValueError:
            logger.info("✓ Unsupported extension handling passed")

        logger.info("Testing invalid CSV return_type handling.")

        try:
            load_file(csv_path, return_type="invalid",)
            raise AssertionError("Expected ValueError")

        except ValueError:
            logger.info("✓ Invalid return_type handling passed")

  
        logger.info("All file_utils integration tests passed successfully.")

    finally:

        shutil.rmtree(test_dir, ignore_errors=True,)
        logger.info("Temporary test directory removed.")