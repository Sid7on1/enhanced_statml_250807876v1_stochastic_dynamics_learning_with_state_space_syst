import logging
import numpy as np
import cv2
import torch
from typing import Tuple, List
from PIL import Image
from torchvision import transforms
from config import Config
from utils import get_logger, get_transforms

class ImagePreprocessor:
    """
    Image preprocessing utilities
    """

    def __init__(self, config: Config):
        """
        Initialize the ImagePreprocessor

        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.transforms = get_transforms(config)

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from a file path

        Args:
            image_path (str): Path to the image file

        Returns:
            np.ndarray: Loaded image
        """
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            self.logger.error(f"Failed to load image: {image_path}")
            raise e

    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize an image to a specified size

        Args:
            image (np.ndarray): Image to resize
            size (Tuple[int, int]): Desired size

        Returns:
            np.ndarray: Resized image
        """
        try:
            image = cv2.resize(image, size)
            return image
        except Exception as e:
            self.logger.error(f"Failed to resize image: {size}")
            raise e

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize an image to the range [0, 1]

        Args:
            image (np.ndarray): Image to normalize

        Returns:
            np.ndarray: Normalized image
        """
        try:
            image = image / 255.0
            return image
        except Exception as e:
            self.logger.error(f"Failed to normalize image")
            raise e

    def apply_transforms(self, image: np.ndarray) -> np.ndarray:
        """
        Apply transforms to an image

        Args:
            image (np.ndarray): Image to transform

        Returns:
            np.ndarray: Transformed image
        """
        try:
            image = self.transforms(image)
            return image
        except Exception as e:
            self.logger.error(f"Failed to apply transforms to image")
            raise e

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess an image

        Args:
            image_path (str): Path to the image file

        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            image = self.load_image(image_path)
            image = self.resize_image(image, self.config.image_size)
            image = self.normalize_image(image)
            image = self.apply_transforms(image)
            return image
        except Exception as e:
            self.logger.error(f"Failed to preprocess image: {image_path}")
            raise e


class Config:
    """
    Configuration object
    """

    def __init__(self):
        """
        Initialize the Config object
        """
        self.image_size = (224, 224)


class Logger:
    """
    Logger class
    """

    def __init__(self, name: str):
        """
        Initialize the Logger

        Args:
            name (str): Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

    def debug(self, message: str):
        """
        Log a debug message

        Args:
            message (str): Message to log
        """
        self.logger.debug(message)

    def info(self, message: str):
        """
        Log an info message

        Args:
            message (str): Message to log
        """
        self.logger.info(message)

    def warning(self, message: str):
        """
        Log a warning message

        Args:
            message (str): Message to log
        """
        self.logger.warning(message)

    def error(self, message: str):
        """
        Log an error message

        Args:
            message (str): Message to log
        """
        self.logger.error(message)


def get_logger(name: str) -> Logger:
    """
    Get a logger instance

    Args:
        name (str): Logger name

    Returns:
        Logger: Logger instance
    """
    return Logger(name)


def get_transforms(config: Config) -> transforms.Compose:
    """
    Get transforms for an image

    Args:
        config (Config): Configuration object

    Returns:
        transforms.Compose: Transforms for an image
    """
    return transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    config = Config()
    preprocessor = ImagePreprocessor(config)
    image_path = "path_to_your_image.jpg"
    image = preprocessor.preprocess_image(image_path)
    print(image.shape)