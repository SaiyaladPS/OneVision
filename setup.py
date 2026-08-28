"""Setuptools hook that includes the existing root-level scan engine in wheels."""

from pathlib import Path
from shutil import copy2

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithScanEngine(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        for filename in ("scan.py", "plate_ocr_torch.py", "pipeline_config.yaml"):
            source = root / filename
            if source.is_file():
                copy2(source, Path(self.build_lib) / filename)


setup(cmdclass={"build_py": BuildPyWithScanEngine})
