# src/eplus_gym/envs/utils.py
import glob
import os
import sys
from typing import Optional, Tuple

def solve_energyplus_install_path() -> Optional[str]:
    """
    Return a directory to add to sys.path so that `import pyenergyplus` succeeds.
    Prefers the install's Python folder (…/Python) where the `pyenergyplus` package lives.
    Looks at ENERGYPLUS_HOME first, then common install locations by OS.
    """
    candidates = []

    # Explicit override
    if (home := os.getenv("ENERGYPLUS_HOME")):
        candidates += [home, os.path.join(home, "Python")]

    # Windows common installs
    if sys.platform.startswith("win"):
        candidates += glob.glob(r"C:\EnergyPlusV*\Python")
        candidates += glob.glob(r"C:\Program Files\EnergyPlus*\Python")

    # macOS
    if sys.platform == "darwin":
        candidates += glob.glob("/Applications/EnergyPlus*/Python")

    # Linux
    if sys.platform.startswith("linux"):
        candidates += glob.glob("/usr/local/EnergyPlus-*/Python")
        candidates += glob.glob("/usr/local/EnergyPlus-*")

    # Return the newest path that actually contains the pyenergyplus package
    def has_pyenergyplus(p: str) -> bool:
        return os.path.exists(os.path.join(p, "pyenergyplus", "__init__.py"))

    for p in sorted(candidates, reverse=True):
        if has_pyenergyplus(p):
            return p
    return None


def try_import_energyplus_api(do_import: bool = True):
    """
    Try to import pyenergyplus. If unavailable, attempt to locate the install and
    extend sys.path. If still unavailable, return (None, None, None) so the package
    can import; the env will raise later when a simulation actually starts.
    """
    # 1) Direct import
    try:
        from pyenergyplus.api import EnergyPlusAPI
        from pyenergyplus.datatransfer import DataExchange
        from pyenergyplus.runtime import Runtime
        return EnergyPlusAPI, DataExchange, Runtime
    except Exception:
        pass

    # 2) Try to resolve install folder and import again
    eplus_path = solve_energyplus_install_path()
    if eplus_path and eplus_path not in sys.path:
        sys.path.append(eplus_path)
        try:
            from pyenergyplus.api import EnergyPlusAPI
            from pyenergyplus.datatransfer import DataExchange
            from pyenergyplus.runtime import Runtime
            return EnergyPlusAPI, DataExchange, Runtime
        except Exception:
            pass

    # 3) Defer failure to runtime (when the env actually constructs the runner)
    return None, None, None


def override(cls):
    """Annotation to document that a method overrides a superclass method."""
    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError(f"{method} does not override any method of {cls}")
        return method
    return check_override
