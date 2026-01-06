from pathlib import Path
from typing import Dict, Tuple

from importlib.resources import files, as_file

import shutil, tempfile
from eplus_gym.envs import assets as assets_pkg

import gymnasium as gym
import numpy as np

from eplus_gym.envs.energyplus import EnergyPlusEnv
from eplus_gym.envs.utils import override


def _asset_to_disk(filename: str) -> str:
    """
    Return a real path to a packaged asset.
    - Verifies the asset exists in eplus_gym.envs.assets
    - Copies (or refreshes) a cached copy
    - Raises a clear error if the asset name is wrong
    """
    pkg_root = files(assets_pkg)

    # 1) verify asset exists in the package and get a real source path
    try:
        res = pkg_root / filename
        with as_file(res) as src_path:
            src_path = Path(src_path)  # real file (even if wheel is zipped)
            if not src_path.exists():
                raise FileNotFoundError
            src_mtime = src_path.stat().st_mtime
    except Exception as e:
        available = []
        try:
            available = [p.name for p in pkg_root.iterdir()]
        except Exception:
            pass
        raise FileNotFoundError(
            f"Asset '{filename}' not found in eplus_gym.envs.assets. "
            f"Available: {available}"
        ) from e

    # 2) place/refresh cached copy
    cache_dir = Path(tempfile.gettempdir()) / "eplus_gym_assets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / filename

    need_copy = (not dst.exists())
    if not need_copy:
        try:
            if Path(dst).stat().st_mtime < src_mtime:
                need_copy = True
        except OSError:
            need_copy = True

    if need_copy:
        shutil.copy(src_path, dst)

    return str(dst)


# --- add near the imports (top of env.py) ---
LOCATION_TO_EPW = {
    "luxembourg": "LUX_LU_Luxembourg.AP.065900_TMYx.2004-2018.epw",
    "usa": "USA_SD_Rapid.City.Rgnl.AP.726620_TMYx.2007-2021.epw",
    "africa": "ZAF_WC_Cape.Town.Portnet.688170_TMYx.2009-2023.epw",
    "portugal": "PRT_LB_Lisboa.Portela.AP.085360_TMYx.2007-2021.epw",
    "brest": "FRA_BT_Brest.Bretagne.AP.071100_TMYx.2009-2023.epw",
    "australia": "AUS_WA_Perth.Metro.946080_TMYx.2007-2021.epw",
}

def _normalize_location(val) -> str:
    return str(val or "Luxembourg").strip().lower()

#%% my agent
    
class AmphitheaterEnv(EnergyPlusEnv):
    '''University amphitheatre environment.

    This environment is based on an actual university amphitheatre in Luxembourg. The building model
    (calibrated against actual energy consumption) of this amphitheatre is available in the same folder.
    The weather file is a typical meteorological year (TMY) weather file.

    HVAC: an AHU with a heating hot water coil, and supply and exhaust air fans.

    Target actuator: supply air temperature setpoint.
    '''  
   
    def __init__(self, env_config=None, new_begin_month=1, new_end_month=12,
             new_begin_day=1, new_end_day=31,
             train=True, w_t=100, w_co2=10, w_dh=10, w_elc=1):
        if env_config is None:
            env_config = {}
        # normalize location early so base class sees the canonical value
        env_config["location"] = _normalize_location(env_config.get("location", "Luxembourg"))
        
        super().__init__(env_config=env_config,
                     new_begin_month=new_begin_month, new_end_month=new_end_month,
                     new_begin_day=new_begin_day, new_end_day=new_end_day,
                     train=train, w_t=w_t, w_co2=w_co2, w_dh=w_dh, w_elc=w_elc)

    
    @override(EnergyPlusEnv)
    def get_weather_file(self) :
        loc = _normalize_location(self.env_config.get("location", "Luxembourg"))
        try:
            epw_name = LOCATION_TO_EPW[loc]
        except KeyError:
            raise ValueError(
                f"Unknown location '{loc}'. Supported locations: {', '.join(sorted(LOCATION_TO_EPW))}"
            )
        return _asset_to_disk(epw_name)
    
    @override(EnergyPlusEnv)
    def get_idf_file(self) :
        return _asset_to_disk("model.idf")

    @override(EnergyPlusEnv)
    def get_observation_space(self) -> gym.Space:
        # observation space:
        # OAT, IAT, cooling setpoint, heating setpoint,number_occupant, district heating, time of the day, Electricity:HVAC, OAT_15, OAT_30, OAT_45, OAT_60,heating_setpoint_15,heating_setpoint_30,heating_setpoint_45,heating_setpoint_60,cooling_setpoint_15,cooling_setpoint_30,cooling_setpoint_45,cooling_setpoint_60
       
        low_obs = np.array([
                           -20,                    #Dewpoint Temperature
                            10,                       #Relative Humidity
                            0,                             #Diffuse Solar Radiation
                            0,                             #Direct Solar Radiation
                            0,                             #Wind Speed
                            0,                             #Wind Direction
                            -10,                   #Drybulb Temperature
                            0.0,                   #occupant
                            16.61111111,                        #HTG HVAC
                            24,                        #CLG HVAC 
                            0.0,                                   #supply air temp 
                            0.0,                                   #flow rate
                            0,                                     #AHU state
                            0.0,                                   #IAT
                            400,                                   #CO2 
                            0.0,                                   #DH
                            0.0,                                   #time
                            0.0                                    #elec 
                            ])                                  
        hig_obs = np.array([
                           30,                         #Dewpoint Temperature
                           100,                    #Relative Humidity
                           500,                    #Diffuse Solar Radiation 
                           1000,               #Direct Solar Radiation
                           20,                         #Wind Speed
                           360,                    #Wind Direction 
                           40.0,               #Drybulb Temperature
                           200,                    #occupant
                           21,                         #HTG HVAC
                           30,                         #CLG HVAC
                           7,                                  #supply air temp
                           7,                                  #flow rate
                           1,                                      #AHU state
                           40.0,                                   #IAT
                           2000,                       #CO2
                           90152325.1800933,                       #DH
                           672,                                    #time 
                           5_037_597.12                            #elec
                           ])                          


        return gym.spaces.Box(low=low_obs, high=hig_obs, dtype=np.float32)

    
    @override(EnergyPlusEnv)
    def get_action_space(self) -> gym.Space:
        
        self.valid_actions = [(i, j, 1) for i in range(8) for j in range(8)] + [(0, 0, 0)]  
        return gym.spaces.Discrete(len(self.valid_actions))
    
    
    

    @override(EnergyPlusEnv)
    def get_variables(self) -> Dict[str, Tuple[str, str]]:
        return {
            # °C(Outdoor Dew Point)
            "odp": ("Site Outdoor Air Dewpoint Temperature", "Environment"),
            # Outdoor Relative Humidity(%)
            "orh": ("Site Outdoor Air Relative Humidity", "Environment"),
            # Diffuse Solar Radiation Rate(w/m2)
            "dsr": ("Site Diffuse Solar Radiation Rate per Area", "Environment"),
            # Direct Solar Radiation Rate(w/m2)
            "dsrr": ("Site Direct Solar Radiation Rate per Area", "Environment"),
            # Wind Speed (m/s)
            "ws": ("Site Wind Speed", "Environment"),
            # Wind Direction (degree)
            "wd": ("Site Wind Direction", "Environment"),
            # °C
            "oat": ("Site Outdoor Air DryBulb Temperature", "Environment"),
            # °C
            "iat": ("Zone Mean Air Temperature", "TZ_Amphitheater"),
            # ppm
            "co2": ("Zone Air CO2 Concentration", "TZ_Amphitheater"),
            # heating setpoint (°C)
            "htg_spt": ("Schedule Value", "HTG HVAC 1 ADJUSTED BY 1.1 F"),
            # cooling setpoint (°C)
            "clg_spt": ("Schedule Value", "CLG HVAC 1 ADJUSTED BY 0 F"),
            # number of occupant (fraction)
            "occ": ("Schedule Value", "Maison Du Savoir Auditorium Occ"),
            # AHU state
            "ahu": ("Schedule Value", "AHUs OnOff"),
            
        }

    @override(EnergyPlusEnv)
    def get_meters(self) -> Dict[str, str]:
        return {
            
            
            "dh": "Heating:DistrictHeatingWater",
            
            "elec": "Electricity:HVAC",
            
        }

    @override(EnergyPlusEnv)
    def get_actuators(self) -> Dict[str, Tuple[str, str, str]]:
        return {
            # supply air temperature setpoint (°C) (2,3,1)
            "sat_spt": ("System Node Setpoint", "Temperature Setpoint", "Node 3"),
            
            
            # Outdoor_Air_Mass_Flow_Rate
            "sat_spt2": ("Outdoor Air Controller", "Air Mass Flow Rate", "CONTROLLER OUTDOOR AIR 1"),
            
            
                          #variable categorie:2   # variable name:3  # key value:1
            "sat_spt3": ("Schedule:Year", "Schedule Value", "AHUS ONOFF"),
            
            
            "sat_spt5": ("Schedule:Year", "Schedule Value", "MAISON DU SAVOIR AUDITORIUM OCC"),
            
            
            "sat_spt6": ("Zone Infiltration", "Air Exchange Flow Rate", "TZ_AMPHITHEATER 189.1-2009 - SECSCHL - AUDITORIUM - CZ4-8 INFILTRATION 77.0 PERCENT REDUCTION"),
            
            
            
        }
    
    
    @override(EnergyPlusEnv)
    def compute_reward(self, obs: Dict[str, float],w_t:float,w_co2:float,w_dh:float,w_elc:float) -> float:
        """
        A more structured reward function that penalizes:
          1. Thermal discomfort (based on comfort band).
          2. CO2 concentration violations.
          3. Normalized energy consumption.
        Weights are introduced to allow the user to tune the relative importance
        of each component.
        """

        # -------------------------
        # 1. Define comfort ranges
        # -------------------------
        # You can adjust these according to your building’s typical comfort policies.
        T_min, T_max = obs["htg_spt"], obs["clg_spt"]        # Example comfort temperature range in °C
        CO2_min, CO2_max = 400.0, 1000.0 # Example healthy CO2 range in ppm

        # Occupant presence factor
        occ = int(obs.get("occ", 0)!=0)

        # -----------------------------------------------------
        # 2. Compute thermal comfort penalty (only if occupied)
        # -----------------------------------------------------
        # If iat is below T_min or above T_max, penalize proportionally to the deviation.
        # This way, being "slightly out" of range is penalized less than "far out".
        iat = obs.get("iat", 22.5)  # default to some mid-range if not present
        if iat < T_min:
            temp_penalty = (T_min - iat)
        elif iat > T_max:
            temp_penalty = (iat - T_max)
        else:
            temp_penalty = 0.0

        # We scale the penalty by occupancy
        # (When unoccupied, there's no occupant to be uncomfortable.)
        temp_penalty *= occ

        # ------------------------------------------------------------------
        # 3. Compute CO2 penalty (again, only if occupied)
        # ------------------------------------------------------------------
        co2 = obs.get("co2", 600.0)  # Some default
        if co2>1000:
            # e.g. penalty grows as we go further below 400 ppm
            co2_penalty = 1
        else :
            co2_penalty = 0

        # Again, scale by occupancy—only penalize when people are present
        co2_penalty *= occ

        # ------------------------------------------------
        # 4. Compute normalized energy penalty
        # ------------------------------------------------
        # Normalize by typical peak or historical max consumption to keep terms of the same order.
        # Make sure these reference values make sense for your particular building/scenario.
        ref_elec = 5_037_597.12     # from your code
        ref_dh = 90152325.1800933      # from your code

        elec = obs.get("elec", 0.0)
        dh   = obs.get("dh",   0.0)

        # Summation of normalized electricity & district heating usage
        energy_penalty = (elec / ref_elec) + (dh / ref_dh)

        # ------------------------------------------------
        # 5. Weight & combine penalties into final reward
        # ------------------------------------------------
        # You can tune these hyperparameters to increase or decrease emphasis on each penalty.
        w_temp = w_t      # weight for temperature discomfort
        w_co2  = w_co2     # weight for CO2 discomfort
        w_en_elec   = w_elc      # weight for energy usage
        w_en_dh=w_dh
        total_penalty = temp_penalty + co2_penalty 

        # By default, reward is the negative of total penalty.
        # That is, the agent is 'rewarded' more when the penalty is lower.
        reward = -total_penalty

        # --------------------------------------
        # (Optional) Additional custom penalties
        # --------------------------------------
        # For example, if the AHU is on (ahu==1) but no electricity usage is measured, 
        # you might suspect a sensor problem or an inefficiency scenario:
        

        return reward
       
    
    def post_process_action(self, action: Tuple[float, float, float]) -> Tuple[float, float, float]:
        actual_range = (15.0, 30.0)
        actual_range1 = (0.3, 5)
        actual_range2=(0,1)
        return (self._rescale(n=action[0], range1=(0, 8 - 1), range2=actual_range), 
                self._rescale(n=action[1], range1=(0, 8 - 1), range2=actual_range1),
                int(self._rescale(n=action[2], range1=(0, 2 - 1), range2=actual_range2)))

    def _rescale(self, n: float, range1: Tuple[float, float], range2: Tuple[float, float]) -> float:
        delta1 = range1[1] - range1[0]
        delta2 = range2[1] - range2[0]
        return (delta2 * (n - range1[0]) / delta1) + range2[0]
    
   
    


