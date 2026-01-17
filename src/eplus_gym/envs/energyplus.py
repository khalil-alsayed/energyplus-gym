

# ==============================================================================
# Original Author: Antoine Galataud (Copyright (c) 2022, MIT License)
# Modifications by: Khalil Al Sayed (Copyright (c) 2026)
# 
# This file is part of research Materials licensed under the 
# Khalil Al Sayed Community Research License Agreement.
# ==============================================================================


import abc
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
import gymnasium as gym
import numpy as np
import pandas as pd
from importlib.resources import files, as_file
from eplus_gym.envs import assets as assets_pkg  # assets/ has an __init__.py
from eplus_gym.envs.utils import try_import_energyplus_api

EnergyPlusAPI, DataExchange, _ = try_import_energyplus_api()


def modify_runperiod_dates(file_path, new_begin_month, new_end_month, 
                           new_begin_day, new_end_day):
    """
    Updates the Begin/End Month and Begin/End Day of Month fields
    in the RunPeriod object of an IDF file. Only lines within the
    RunPeriod object are modified; other objects remain unchanged.

    :param file_path: Path to the IDF file
    :param new_begin_month: (int) New Begin Month
    :param new_end_month:   (int) New End Month
    :param new_begin_day:   (int) New Begin Day of Month
    :param new_end_day:     (int) New End Day of Month
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_run_period = False

    for i, line in enumerate(lines):
        # 1. Detect the start of the RunPeriod object
        if line.strip().startswith("RunPeriod,"):
            in_run_period = True

        if in_run_period:
            # 2. BEGIN MONTH
            if '!- Begin Month' in line:
                parts = line.split(',')
                if len(parts) > 0:
                    idx = parts[0].rfind(' ')
                    if idx != -1:
                        parts[0] = parts[0][:idx+1] + str(new_begin_month)
                    else:
                        parts[0] = str(new_begin_month)
                    lines[i] = ','.join(parts)

            # 3. END MONTH
            elif '!- End Month' in line:
                parts = line.split(',')
                if len(parts) > 0:
                    idx = parts[0].rfind(' ')
                    if idx != -1:
                        parts[0] = parts[0][:idx+1] + str(new_end_month)
                    else:
                        parts[0] = str(new_end_month)
                    lines[i] = ','.join(parts)

            # 4. BEGIN DAY OF MONTH
            elif '!- Begin Day of Month' in line:
                parts = line.split(',')
                if len(parts) > 0:
                    idx = parts[0].rfind(' ')
                    if idx != -1:
                        parts[0] = parts[0][:idx+1] + str(new_begin_day)
                    else:
                        parts[0] = str(new_begin_day)
                    lines[i] = ','.join(parts)

            # 5. END DAY OF MONTH
            elif '!- End Day of Month' in line:
                parts = line.split(',')
                if len(parts) > 0:
                    idx = parts[0].rfind(' ')
                    if idx != -1:
                        parts[0] = parts[0][:idx+1] + str(new_end_day)
                    else:
                        parts[0] = str(new_end_day)
                    lines[i] = ','.join(parts)

            # 6. Detect the end of the RunPeriod object 
            #    (when we hit the semicolon)
            if ';' in line:
                in_run_period = False

    # 7. Write the modifications back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
        
def _prepare_idf_with_runperiod(original_idf: Union[str, Path],
                                 output_dir: Union[str, Path],
                                 new_begin_month: int, new_end_month: int,
                                 new_begin_day: int,   new_end_day: int) -> str:
    """
    Copy the input IDF next to the simulation outputs and patch its RunPeriod.
    Returns the path to the patched copy.
    """
    src = Path(original_idf)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dst = out / f"{src.stem}_runperiod_{new_begin_month:02d}{new_begin_day:02d}-{new_end_month:02d}{new_end_day:02d}{src.suffix}"

    # Always copy from source so we start from a clean base
    shutil.copy(src, dst)
    modify_runperiod_dates(str(dst), new_begin_month, new_end_month, new_begin_day, new_end_day)
    return str(dst)
     

# --- Project root discovery -----------------------------------------------------

def _find_project_root() -> Path:
    """
    Return the repository root directory (the folder that contains 'eplus_outputs' and 'pyproject.toml').

    Priority:
      1) EPLUS_GYM_PROJECT_ROOT env var (explicit, absolute or relative).
      2) If running from a src-layout …/src/eplus_gym/… or …/src/rleplus/…, go to parent of 'src'.
      3) Walk up from this file looking for repo markers (pyproject.toml or .git) or a known root name.
      4) If still not found, walk up from CWD for the same markers.
      5) Fallback: CWD (site-packages case).
    """
    # 1) explicit override
    env_root = os.getenv("EPLUS_GYM_P​ROJECT_ROOT")
    if env_root:
        p = (Path.cwd() / env_root).expanduser().resolve()
        if p.exists():
            return p

    here = Path(__file__).resolve()

    # 2) src-layout heuristic: …/<root>/src/(eplus_gym|rleplus)/…
    parents = [p.name.lower() for p in here.parents]
    try:
        for pkg_name in ("eplus_gym", "rleplus"):
            if pkg_name in parents:
                i = parents.index(pkg_name)
                if i + 1 < len(parents) and parents[i + 1] == "src":
                    root = here.parents[i + 2]          # parent of 'src' => repo root
                    if (root / "pyproject.toml").exists() or (root / ".git").exists():
                        return root.resolve()
    except Exception:
        pass

    markers = ("pyproject.toml", ".git")
    root_names = {"energyplus-gym", "rleplus-gym"}  # accept either repo name

    def walk_up(start: Path) -> Path | None:
        for d in (start.resolve(), *start.resolve().parents):
            try:
                names = {p.name.lower() for p in d.iterdir()}
            except Exception:
                continue
            if any((d / m).exists() for m in markers) or d.name.lower() in root_names:
                return d
        return None

    # 3) walk up from this file
    r = walk_up(here)
    if r:
        return r

    # 4) walk up from current working directory
    r = walk_up(Path.cwd())
    if r:
        return r

    # 5) fallback: CWD (e.g., running from site-packages)
    return Path.cwd().resolve()


# --- Output directory resolution ------------------------------------------------

def _resolve_output_dir(value: str | os.PathLike | None,
                        default_name: str = "eplus_outputs") -> str:
    """
    Resolve the output directory under the *project root* by default.

      - value is None/''  ->  <project_root>/<default_name>
      - value is relative ->  <project_root>/<value>
      - value is absolute ->  use as-is

    Ensures the directory exists and returns an absolute, normalized string path.
    """
    proj_root = _find_project_root()

    if value:
        raw = (Path.cwd() / str(value)).expanduser().resolve() if not os.path.isabs(str(value)) \
              else Path(str(value)).expand_absolute() if hasattr(Path('.'), "expand_absolute") else Path(str(value)).resolve()
        p = raw if raw.is_absolute() else (proj_root / raw).resolve()
    else:
        p = (proj_root / default_name).resolve()

    p.mkdir(parents=True, exist_ok=True)
    return str(p)


'''
def _resolve_output_dir(value, default_name="eplus_outputs") -> str:
    """
    Resolve the output directory:
      - If value is None or "", use ./eplus_outputs relative to the current working directory.
      - If value is relative, make it relative to the current working directory.
      - If value is absolute, use it as-is.
    In all cases, create the directory if it doesn't exist.
    Returns a normalized absolute path as a string.
    """
    if value:
        raw = os.path.expanduser(os.path.expandvars(str(value)))
        p = Path(raw)
        if not p.is_absolute():
            p = Path.cwd() / p
    else:
        p = Path.cwd() / default_name

    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())

'''

def _normalize_location(val) -> str:
    return str(val or "Luxembourg").strip().lower()

CSV_BY_LOCATION = {
    "luxembourg": "Luxembourg.csv",
    "usa": "USA.csv",
    "africa": "africa.csv",
    "portugal": "portugal.csv",
    "brest": "brest.csv",
    "australia": "australia.csv",
}
# --------------------------------------------------------------


#%% my agent
def parse_custom_datetime(s):
    """
    Parse a string of the form "MM/DD HH:MM:SS".
    If the time is "24:00:00", treat it as 86400 seconds (i.e. the end of that day)
    without rolling the date over.
    Returns a tuple: (parsed_date, seconds_into_day)
    where parsed_date is a Timestamp (dummy year 1900) and seconds_into_day is an integer.
    """
    try:
        date_part, time_part = s.split(" ")
    except Exception:
        return None, None  # In case of a parsing error
    
    # Parse the date part using a dummy year (1900)
    dt_date = pd.to_datetime("1900/" + date_part, format='%Y/%m/%d', errors='coerce')
    if pd.isna(dt_date):
        return None, None

    # Handle the time part:
    if time_part == "24:00:00":
        seconds = 86400
    else:
        try:
            # Convert time string to seconds from midnight
            h, m, s_ = map(int, time_part.split(":"))
            seconds = h * 3600 + m * 60 + s_
        except Exception:
            return None, None
    return dt_date, seconds



@dataclass
class RunnerConfig:
    '''Configuration for the runner.'''

    # Path to the weather file (.epw)
    epw: Union[Path, str]
    # Path to the IDF file
    idf: Union[Path, str]
    # Path to the output directory
    output: Union[Path, str]
    # EnergyPlus variables to request
    variables: Dict[str, Tuple[str, str]]
    # EnergyPlus meters to request
    meters: Dict[str, str]
    # EnergyPlus actuators to actuate
    actuators: Dict[str, Tuple[str, str, str]]
    # Generate eplusout.csv at end of simulation
    csv: bool = False
    # In verbose mode, EnergyPlus will print to stdout
    verbose: bool = False
    # EnergyPlus timestep duration, in fractional hour. Default is 0.25 (15 minutes)
    eplus_timestep_duration: float = 0.25

    def __post_init__(self):
        self.epw = str(self.epw)
        self.idf = str(self.idf)
        self.output = str(self.output)
        # Make sure the output directory exists
        Path(self.output).mkdir(parents=True, exist_ok=True)

        # check provided paths exist
        for path in [self.epw, self.idf]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        # check variables, meters and actuators are not empty
        for name, data in [("variables/meters", {**self.variables, **self.meters}), ("actuators", self.actuators)]:
            if len(data) == 0:
                raise ValueError(f"No {name} provided")

        assert self.eplus_timestep_duration > 0.0, "E+ timestep duration must be > 0.0"


class EnergyPlusRunner:
    '''EnergyPlus simulation runner.

    This class is responsible for running EnergyPlus in a separate thread and to interact
    with it through its API.
    '''

    def __init__(self, episode: int, obs_queue: Queue, act_queue: Queue, runner_config: RunnerConfig) -> None:
        self.episode = episode
        self.runner_config = runner_config
        self.verbose = self.runner_config.verbose

        self.obs_queue = obs_queue
        self.act_queue = act_queue
        # protect act_queue from concurrent access that can happen at end of simulation
        self.act_queue_mutex = threading.Lock()

        self.energyplus_api = EnergyPlusAPI()
        self.x: DataExchange = self.energyplus_api.exchange
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: Dict[str, Any] = {}
        self.initialized = False
        self.progress_value: int = 0
        self.simulation_complete = False
        # Zone timestep duration, in fractional hour. Default is 15 minutes
        # Make sure to set this value to reflect your simulation timestep (ie 4 steps per hour in IDF = 0.25)
        self.zone_timestep_duration = self.runner_config.eplus_timestep_duration

        # below is declaration of variables, meters and actuators
        # this simulation will interact with
        self.variables = runner_config.variables
        self.var_handles: Dict[str, int] = {}

        self.meters = runner_config.meters
        self.meter_handles: Dict[str, int] = {}

        self.actuators = runner_config.actuators
        self.actuator_handles: Dict[str, int] = {}
        self.last_action = (0.0, 0.0, 0.0)

    def start(self) -> None:
        self.energyplus_state = self.energyplus_api.state_manager.new_state()
        runtime = self.energyplus_api.runtime

        # register callback used to track simulation progress
        def _report_progress(progress: int) -> None:
            self.progress_value = progress
            if self.verbose:
                print(f"Simulation progress: {self.progress_value}%")

        runtime.callback_progress(self.energyplus_state, _report_progress)

        runtime.set_console_output_status(self.energyplus_state, self.verbose)

        # register callback used to collect observations
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)

        # register callback used to send actions
        runtime.callback_after_predictor_after_hvac_managers(self.energyplus_state, self._send_actions)

        # run EnergyPlus in a non-blocking way
        def _run_energyplus(rn, cmd_args, state, results):
            if self.verbose:
                print(f"running EnergyPlus with args: {cmd_args}")

            # start simulation
            results["exit_code"] = rn.run_energyplus(state, cmd_args)

            if not self.simulation_complete:
                # free consumers from waiting
                self.obs_queue.put(None)
                self.act_queue.put(None)
                self.stop()

        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=(self.energyplus_api.runtime, self.make_eplus_args(), self.energyplus_state, self.sim_results),
        )
        self.energyplus_exec_thread.start()


    '''
    def stop(self) -> None:
        if not self.simulation_complete:
            self.simulation_complete = True
            self._flush_queues()
            self.energyplus_exec_thread.join()
            self.energyplus_exec_thread = None
            self.energyplus_api.runtime.clear_callbacks()
            self.energyplus_api.state_manager.delete_state(self.energyplus_state)
    '''
            
    def stop(self) -> None:
        # make stop idempotent
        if getattr(self, "simulation_complete", False):
            return

        self.simulation_complete = True
        self._flush_queues()

        t = getattr(self, "energyplus_exec_thread", None)

        # Only join if we're NOT in the worker thread
        if t is not None and t.is_alive() and threading.current_thread() is not t:
            t.join()
            self.energyplus_exec_thread = None

        # Do the rest of your cleanup
        self.energyplus_api.runtime.clear_callbacks()
        self.energyplus_api.state_manager.delete_state(self.energyplus_state)



            
        

    def failed(self) -> bool:
        return self.sim_results.get("exit_code", -1) > 0

    def make_eplus_args(self) -> List[str]:
        '''Make command line arguments to pass to EnergyPlus.'''
        eplus_args = ["-r"] if self.runner_config.csv else []
        eplus_args += [
            "-w",
            self.runner_config.epw,
            "-d",
            f"{self.runner_config.output}",
            #f"{self.runner_config.output}/episode-{self.episode:08}-{os.getpid():05}",
            self.runner_config.idf,
        ]
        return eplus_args

    def init_exchange(self, default_action: Tuple[float, float]) -> Dict[str, float]:
        
        self.last_action = default_action
        
        self.act_queue.put(default_action)
        return self.obs_queue.get()

    def _collect_obs(self, state_argument) -> None:
        '''EnergyPlus callback that collects output variables/meters values and enqueue them.'''
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        self.next_obs = {
            **{key: self.x.get_variable_value(state_argument, handle) for key, handle in self.var_handles.items()},
            **{key: self.x.get_meter_value(state_argument, handle) for key, handle in self.meter_handles.items()},
        }
        self.obs_queue.put(self.next_obs)

    def _send_actions(self, state_argument):
        '''EnergyPlus callback that sets actuator value from last decided action.'''
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        # E+ has zone and system timesteps, a zone timestep can be made of several system timesteps
        # (number varies on each iteration). We should send actions at least once per zone timestep, so we can
        # resend the last action if we are iterating over system timesteps, but we need to wait for a new action
        # when moving from one zone timestep to another.
        sys_timestep_duration = self.x.system_time_step(state_argument)
        if sys_timestep_duration < self.zone_timestep_duration and self.act_queue.empty():
            self.act_queue.put(self.last_action)

        # wait for next action
        with self.act_queue_mutex:
            if self.simulation_complete:
                return
            next_action = self.act_queue.get()

        # end of simulation
        if next_action is None:
            self.simulation_complete = True
            return

        assert isinstance(next_action, tuple)

        # keep last action to resend it if needed (see above)
        self.last_action = next_action

        self.x.set_actuator_value(
            state=state_argument, actuator_handle=self.actuator_handles["sat_spt"], actuator_value=next_action[0]
        )
        self.x.set_actuator_value(
            state=state_argument, actuator_handle=self.actuator_handles["sat_spt2"], actuator_value=next_action[1]
        )
        self.x.set_actuator_value(
            state=state_argument, actuator_handle=self.actuator_handles["sat_spt3"], actuator_value=next_action[2]
        )
        
        self.x.set_actuator_value(
            state=state_argument, actuator_handle=self.actuator_handles["sat_spt5"], actuator_value=next_action[3]
        )
        
        self.x.set_actuator_value(
            state=state_argument, actuator_handle=self.actuator_handles["sat_spt6"], actuator_value=0.5
        )
        
        
        
        

    def _init_callback(self, state_argument) -> bool:
        '''Initialize EnergyPlus handles and checks if simulation runtime is ready.'''
        self.initialized = self._init_handles(state_argument) and not self.x.warmup_flag(state_argument)
        return self.initialized

    def _init_handles(self, state_argument):
        '''Initialize sensors/actuators handles to interact with during simulation.'''
        if not self.initialized:
            if not self.x.api_data_fully_ready(state_argument):
                return False

            self.var_handles = {
                key: self.x.get_variable_handle(state_argument, *var) for key, var in self.variables.items()
            }

            self.meter_handles = {
                key: self.x.get_meter_handle(state_argument, meter) for key, meter in self.meters.items()
            }

            self.actuator_handles = {
                key: self.x.get_actuator_handle(state_argument, *actuator) for key, actuator in self.actuators.items()
            }

            for handles in [self.var_handles, self.meter_handles, self.actuator_handles]:
                if any([v == -1 for v in handles.values()]):
                    available_data = self.x.list_available_api_data_csv(state_argument).decode("utf-8")
                    raise RuntimeError(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> variables: {self.var_handles}\n"
                        f"> meters: {self.meter_handles}\n"
                        f"> actuators: {self.actuator_handles}\n"
                        f"> available E+ API data: {available_data}"
                    )

            self.initialized = True

        return True

    def _flush_queues(self):
        # release waiting threads (if any)
        if self.act_queue.empty():
            self.act_queue.put(None)

        while not self.obs_queue.empty():
            self.obs_queue.get()

        # flush actions queue after last callback was called
        with self.act_queue_mutex:
            while not self.act_queue.empty():
                self.act_queue.get()
                
def _read_csv_from_assets(filename: str) -> pd.DataFrame:
    """
    Load a CSV shipped with the package (under rleplus/envs/assets).
    Works from wheels (zipped) by materializing a real file path.
    """
    res = files(assets_pkg) / filename
    with as_file(res) as p:
        return pd.read_csv(p)    
    
class EnergyPlusEnv(gym.Env, metaclass=abc.ABCMeta):
    '''Base, abstract EnergyPlus gym environment.

    This class implements the OpenAI gym (now gymnasium) API. It must be subclassed to
    implement the actual environment.
    '''

    def __init__(self, env_config: Dict[str, Any], new_begin_month, new_end_month, new_begin_day, new_end_day,train=True,w_t=100, w_co2=10,w_dh=10,w_elc=1):
        self.spec = gym.envs.registration.EnvSpec(f"{self.__class__.__name__}")
        self.train=train
        self.env_config = env_config
        self.episode = -1
        self.timestep = 0
        self.score=0
        self.observation_space = self.get_observation_space()
        self.last_obs = {}
        self.w_t=w_t
        self.w_co2=w_co2
        self.w_dh=w_dh
        self.w_elc=w_elc
        self.action_space = self.get_action_space()
        #self.default_action = self.post_process_action(self.action_space.sample())
        #self.default_action= self.post_process_action((self.action_space.sample()[0][0],self.action_space.sample()[0][1],self.action_space.sample()[1]))    SAC
        self.default_action= self.post_process_action(self.valid_actions[self.action_space.sample()])
        self.default_action=list(self.default_action)
        self.default_action.append(0)
        self.default_action=tuple(self.default_action)
        
        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None
        
        # Resolve / default the output directory (and ensure it exists)
        resolved_output = _resolve_output_dir(self.env_config.get("output"))
        self.env_config["output"] = resolved_output  # keep the resolved path in config for later use

        # Resolve weather + IDF, then create a patched copy of the IDF with the requested RunPeriod
        weather_file = self.get_weather_file()
        original_idf = self.get_idf_file()
        patched_idf = _prepare_idf_with_runperiod(
            original_idf, resolved_output,
            new_begin_month, new_end_month, new_begin_day, new_end_day
        )
        self.runner_config = RunnerConfig(
            epw=weather_file,
            idf=patched_idf,
            output=resolved_output,
            variables=self.get_variables(),
            meters=self.get_meters(),
            actuators=self.get_actuators(),
            csv=self.env_config.get("csv", True),
            verbose=self.env_config.get("verbose", False),
            eplus_timestep_duration=self.env_config.get("eplus_timestep_duration", 0.25),
        )
                
        # Choose CSV name: allow an override via env_config["occupancy_csv"],
        # otherwise select a default based on env_config["location"].
        loc_key = _normalize_location(self.env_config.get("location", "Luxembourg"))
        default_csv = CSV_BY_LOCATION.get(loc_key, "Luxembourg.csv")  # fallback keeps backward-compat

        csv_name = self.env_config.get("occupancy_csv", default_csv)

        # Read from an absolute path if given; otherwise load the packaged asset
        if os.path.isabs(csv_name) and os.path.exists(csv_name):
            df = pd.read_csv(csv_name)
        else:
            df = _read_csv_from_assets(csv_name)




        # 1) Normalize whitespace if needed (strip leading/trailing and replace multiple spaces)
        #    If your data is already clean, you can skip the replace step to improve performance further.
        df['Date/Time'] = (
            df['Date/Time']
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )

        # 2) Split into date part (MM/DD) and time part (HH:MM:SS)
        #    expand=True returns a DataFrame with two columns: [0] for date_part, [1] for time_part
        dt_split = df['Date/Time'].str.split(' ', n=1, expand=True)
        dt_split.columns = ['date_part', 'time_part']

        # 3) Parse the date part by prepending a dummy year (1900).
        #    This gives a proper Timestamp we can use to compute day offsets.
        dt_split['ParsedDate'] = pd.to_datetime(
            '1900/' + dt_split['date_part'],
            format='%Y/%m/%d',
            errors='coerce'
        )

        # 4) Convert the time part to seconds since midnight in a vectorized way.
        #    First, handle standard times (HH:MM:SS) by extracting H, M, S.
        #    Then fix rows with "24:00:00" to 86400 seconds.
        #    (This keeps it on the same day, which is not standard but matches your requirement.)
        time_df = dt_split['time_part'].str.extract(
            r'(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})'
        )
        # Convert each captured group to integer
        # or int; float in case of parsing edge issues
        time_df = time_df.astype(float)
        dt_split['SecondsIntoDay'] = (
            time_df['h'] * 3600 + time_df['m'] * 60 + time_df['s']
        )

        # Handle "24:00:00" by assigning 86400 seconds
        mask_24 = dt_split['time_part'] == '24:00:00'
        dt_split.loc[mask_24, 'SecondsIntoDay'] = 86400

        # 5) Compute "DayNumber" relative to 1900-01-01, then "CompositeSeconds".
        dt_split['DayNumber'] = (
            dt_split['ParsedDate'] - pd.Timestamp('1900-01-01')
        ).dt.days
        dt_split['CompositeSeconds'] = (
            dt_split['DayNumber'] * 86400 + dt_split['SecondsIntoDay']
        )

        # 6) Join the parsed columns back to your original DataFrame if needed
        #    (or we can just keep them separate). Here we join so we can filter easily.
        df = df.join(
            dt_split[['ParsedDate', 'SecondsIntoDay', 'DayNumber', 'CompositeSeconds']])

        # 7) Prepare boundaries for filtering
        #    Begin boundary: new_begin_month/new_begin_day at 00:15:00
        begin_date = pd.to_datetime(
            f'1900/{new_begin_month:02d}/{new_begin_day:02d}', format='%Y/%m/%d')
        begin_seconds = 15 * 60  # 00:15:00 is 900 seconds
        begin_day_number = (begin_date - pd.Timestamp('1900-01-01')).days
        begin_composite = begin_day_number * 86400 + begin_seconds

        #    End boundary: new_end_month/new_end_day at 24:00:00 -> 86400 seconds
        end_date = pd.to_datetime(
            f'1900/{new_end_month:02d}/{new_end_day:02d}', format='%Y/%m/%d')
        end_seconds = 86400
        end_day_number = (end_date - pd.Timestamp('1900-01-01')).days
        end_composite = end_day_number * 86400 + end_seconds

        # 8) Filter the DataFrame using a single vectorized condition
        df_filtered = df[
            (df['CompositeSeconds'] >= begin_composite)
            & (df['CompositeSeconds'] <= end_composite)
        ].copy()

        # 9) (Optional) Drop helper columns
        df_filtered.drop(
            columns=['ParsedDate', 'SecondsIntoDay',
                     'DayNumber', 'CompositeSeconds'],
            inplace=True,
            errors='ignore'
        )

        # 10) Create a new DataFrame with 4 rows of zeros and concatenate
        zeros_df = pd.DataFrame(0, index=range(4), columns=df_filtered.columns)
        self.temp = pd.concat([df_filtered, zeros_df], ignore_index=True)
        
   

    @abc.abstractmethod
    def get_weather_file(self) -> Union[Path, str]:
        '''Returns the path to a valid weather file (.epw).

        This method can be used to randomize training data by providing different weather
        files. It's called on each reset()
        '''

    @abc.abstractmethod
    def get_idf_file(self) -> Union[Path, str]:
        '''Returns the path to a valid IDF file.'''

    @abc.abstractmethod
    def get_observation_space(self) -> gym.Space:
        '''Returns the observation space of the environment.'''

    @abc.abstractmethod
    def get_action_space(self) -> gym.Space:
        '''Returns the action space of the environment.'''

    @abc.abstractmethod
    def compute_reward(self, obs: Dict[str, float],w_t:float,w_co2:float,w_dh:float,w_elc:float) -> float:
        '''Computes the reward for the given observation.'''

    @abc.abstractmethod
    def get_variables(self) -> Dict[str, Tuple[str, str]]:
        '''Returns the variables to track during simulation.'''

    @abc.abstractmethod
    def get_meters(self) -> Dict[str, str]:
        '''Returns the meters to track during simulation.'''

    @abc.abstractmethod
    def get_actuators(self) -> Dict[str, Tuple[str, str, str]]:
        '''Returns the actuators to control during simulation.'''

    def post_process_action(self, action: Union[float, List[float]]) -> Union[float, List[float]]:
        '''Post-processes the action(s) before sending it to EnergyPlus.

        This method can be used to implement constraints on the actions, for example.
        Default implementation returns the action unchanged.
        '''
        return action

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.episode += 1
        self.timestep = 0
        self.last_obs = self.observation_space.sample()
        self.score=0
        self.violation_counter=0
        #self.previous_action=-1
        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

        # observations and actions queues for flow control
        # queues have a default max size of 1
        # as only 1 E+ timestep is processed at a time
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)

        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
            runner_config=self.runner_config,
        )
        self.energyplus_runner.start()
                                      
        # wait until E+ is ready.
        self.last_obs = obs = self.energyplus_runner.init_exchange(default_action=self.default_action)
        #date=str(self.month).zfill(2) + "/" + str(self.day).zfill(2)
        return np.array(  # Sequential Data (20 elements)
            # Dewpoint Temperature (odp)
            [obs["odp"]]
            +

            # Relative Humidity (orh)
            [obs["orh"]]
            + 

            # Diffuse Solar Radiation (dsr)
            [obs["dsr"]]
            + 

            # Direct Solar Radiation (dsrr)
            [obs["dsrr"]]
            + 

            # Wind Speed (ws)
            [obs["ws"]]
            + 

            # Wind Direction (wd)
            [obs["wd"]]
            + 

            # Drybulb Temperature (oat)
            [obs["oat"]]
            + 

            # occupant (occ)
            [obs["occ"]]
            + 

            # HTG HVAC (htg_spt)
            [obs["htg_spt"]]
            + 

            # CLG HVAC (clg_spt)
            [obs["clg_spt"]]
            + 
            [0] +  # supply air temp
            [0] +  # flow rate
            [obs["ahu"]] +  # AHU state
            [obs["iat"]] +  # IAT
            [obs["co2"]] +  # CO2
            [obs["dh"]] +  # DH
            [self.timestep % 672] +  # time
            [0],  # elec
            dtype='float32'), {}  # new


    def step(self, action):
        action = self.valid_actions[action]
        self.timestep += 1
        terminated = False
        truncated = False
        # in case of training in complete month----------------------------------------
        
        if self.timestep%672==0 and self.train==True:
            truncated = True
          
        #-----------------------------------------------------------
        # check for simulation errors
        if self.energyplus_runner.failed():
            raise RuntimeError(f"EnergyPlus failed with {self.energyplus_runner.sim_results['exit_code']}")

        # simulation_complete is likely to happen after last env step()
        # is called, hence leading to waiting on queue for a timeout
        if self.energyplus_runner.simulation_complete:
            terminated = True
            obs = self.last_obs
        else:
            # post-process action
            action_to_apply = self.post_process_action(action)
            # Enqueue action (sent to EnergyPlus through dedicated callback)
            # then wait to get next observation.
            # Timeout is set to 2s to handle end of simulation cases, which happens async
            # and materializes by worker thread waiting on this queue (EnergyPlus callback
            # not consuming anymore).
            # Timeout value can be increased if E+ timestep takes longer
            timeout = 2
            try:
                action_to_apply=list(action_to_apply)
                action_to_apply.append(self.temp['occupant'].iloc[self.timestep])
                #action_to_apply.append(self.temp[str(self.month).zfill(2) + "/" + str(self.day).zfill(2), 'occupant'].iloc[(self.timestep+1)%96])
                action_to_apply=tuple(action_to_apply)
                self.act_queue.put(action_to_apply, timeout=timeout)
                obs = self.obs_queue.get(timeout=timeout)
            except (Full, Empty):
                obs = None
                pass

            # obs can be None if E+ simulation is complete
            # this materializes by either an empty queue or a None value received from queue
            if obs is None:
                terminated = True
                obs = self.last_obs
            else:
                self.last_obs = obs
       
        reward = self.compute_reward(obs,w_t=self.w_t,w_co2=self.w_co2,w_dh=self.w_dh,w_elc=self.w_elc) -(action[0]/7)*action[2]  -(action[1]/7)*action[2] -action[2]/5
           
        
        
        obs["occ"] = obs["occ"]*400
        obs_vec = np.array(
            # Sequential Data 
            # Dewpoint Temperature (odp)
            [obs["odp"]]
            +

            # Relative Humidity (orh)
            [obs["orh"]]
            + 

            # Diffuse Solar Radiation (dsr)
            [obs["dsr"]]
            + 

            # Direct Solar Radiation (dsrr)
            [obs["dsrr"]]
            + 

            # Wind Speed (ws)
            [obs["ws"]]
            + 

            # Wind Direction (wd)
            [obs["wd"]]
            + 

            # Drybulb Temperature (oat)
            [obs["oat"]]
            + 

            # occupant (occ)
            [obs["occ"]]
            + 

            # HTG HVAC (htg_spt)
            [obs["htg_spt"]]
            + 

            # CLG HVAC (clg_spt)
            [obs["clg_spt"]]
            + 
            # non-Sequential Data 
            [action[0]]+                                                 #supply air temp
            [action[1]]+                                                 #flow rate
            [obs["ahu"]]+                                                #AHU state
            [obs["iat"]]+                                                #IAT
            [obs["co2"]]+                                                #CO2
            [obs["dh"]]+                                                 #DH
            [self.timestep%672]+                                          #time
            [obs["elec"]],                                               #elec
            dtype='float32')
        
        
        return obs_vec, reward, terminated, truncated, {}  # new

    def close(self):
        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

    def render(self, mode="human"):
        pass
    
   






















