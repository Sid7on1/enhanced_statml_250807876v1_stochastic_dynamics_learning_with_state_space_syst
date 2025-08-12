import os
import sys
import logging
import pkg_resources
from setuptools import setup, find_packages
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("setup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class PackageSetup:
    def __init__(self):
        self.name = "computer_vision"
        self.version = "1.0.0"
        self.author = "Your Name"
        self.author_email = "your@email.com"
        self.description = "Package installation setup for computer vision project"
        self.dependencies = self.get_dependencies()
        self.requirements = self.get_requirements()

    def get_dependencies(self) -> Dict[str, List[str]]:
        """Get dependencies from requirements.txt"""
        try:
            with open("requirements.txt", "r") as f:
                dependencies = {}
                for line in f.readlines():
                    if line.startswith("#"):
                        continue
                    package, version = line.strip().split("==")
                    dependencies[package] = [version]
                return dependencies
        except FileNotFoundError:
            logging.error("requirements.txt not found")
            sys.exit(1)

    def get_requirements(self) -> List[str]:
        """Get requirements from requirements.txt"""
        try:
            with open("requirements.txt", "r") as f:
                requirements = []
                for line in f.readlines():
                    if line.startswith("#"):
                        continue
                    package, version = line.strip().split("==")
                    requirements.append(f"{package}=={version}")
                return requirements
        except FileNotFoundError:
            logging.error("requirements.txt not found")
            sys.exit(1)

    def setup_package(self):
        """Setup package"""
        setup(
            name=self.name,
            version=self.version,
            author=self.author,
            author_email=self.author_email,
            description=self.description,
            packages=find_packages(),
            install_requires=self.requirements,
            include_package_data=True,
            zip_safe=False
        )

    def install_dependencies(self):
        """Install dependencies"""
        try:
            os.system("pip install -r requirements.txt")
        except Exception as e:
            logging.error(f"Failed to install dependencies: {e}")
            sys.exit(1)

def main():
    setup_package = PackageSetup()
    setup_package.install_dependencies()
    setup_package.setup_package()

if __name__ == "__main__":
    main()