import os
import pandas as pd
from utils.logger import logger
from utils.custom_exception import CustomException
import yaml


def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path.")

        with open(file_path, 'r') as yaml_file:
            config =  yaml.safe_load(yaml_file)
            logger.info("Successfully read the yaml file.")
            return config
        
    except Exception as e:
        logger.error("Error while reading YAML File.")
        raise CustomException("Failed to read YAML File" ,e)