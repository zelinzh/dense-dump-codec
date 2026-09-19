"""Include command helpers in wheels without duplicating source files in Git."""

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithCommands(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "dense_dump_codec" / "_runtime" / "scripts"
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(Path("scripts").glob("*.py")):
            shutil.copy2(source, destination / source.name)


setup(cmdclass={"build_py": BuildWithCommands})
