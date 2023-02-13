# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]


# setup.py
setup(
    name="chatbot",
    version=0.1,
    description="Chatbot for movies",
    author="Anirudh Iyer",
    author_email="anirudh99iyer@gmail.com",
    python_requires=">=3.7",
    install_requires=[required_packages],
)


