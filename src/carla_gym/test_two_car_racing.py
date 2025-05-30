import gymnasium as gym
import numpy as np
import time

import torch

from gym_carla.controllers.barc_lmpc import LMPCWrapper
from gym_carla.controllers.barc_mpcc_conv import MPCCConvWrapper
from mpclab_common.track import get_track
from loguru import logger

from gym_carla.controllers.barc_pid import PIDWrapper
from gym_carla.controllers.barc_pid_ref_tracking import PIDRacelineFollowerWrapper
from torch.distributions import Normal
from mpcexp.controllers import AttackerBarcWrapper, DefenderBarcWrapper, KinematicBicycleBarcWrapper
from mpcexp.utils.utils_fun import save_sim_data
import signal
import sys
import datetime

save_data = True
log_data = []

def save_and_exit(signum, frame):
    print(f"Received signal {signum}. Saving data...")
    # ----- Save Data -----
    if save_data:
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"autosim_{time_str}_RA-GTP_fast_barc_gym"
        save_sim_data(log_data, format=".pkl", filename=file_name)
        print(f"Simulation data saved as {file_name}.pkl")
    sys.exit(0)

# Register signals
signal.signal(signal.SIGINT, save_and_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, save_and_exit)  # Termination

def main(seed=0):
    """
    Test script for the two-car racing environment.
    Uses LMPCWrapper for the ego vehicle and PIDWrapper for the opponent.
    """
    dt = 0.1
    dt_sim = 0.01
    t0 = 0
    rng = np.random.default_rng()
    track_name = 'L_track_barc'

    # Create the two-car racing environment
    controller_type = [PIDRacelineFollowerWrapper, PIDRacelineFollowerWrapper]
    delay_steps = 1
    ego_controller = AttackerBarcWrapper(experiment="RA-GTP", delay_steps=delay_steps)
    opponent_controller = DefenderBarcWrapper(delay_steps=delay_steps)
    # kinBicycleMPC = KinematicBicycleBarcWrapper(dt=dt, t0=t0, track_obj=get_track(track_name))

    env = gym.make('barc-v1',
                   # opponent=opponent_controller,
                   track_name=track_name,
                   t0=t0, dt=dt, dt_sim=dt_sim,
                   do_render=True,
                   enable_camera=False,
                   discrete_action=False)

    # Bind the controllers to the environment
    env.unwrapped.bind_controller(ego_controller)

    # Reset the environment
    ob, info = env.reset(seed=seed, options={'spawning': 'fixed'})

    # Reset the controllers
    ego_controller.reset(vehicle_state=info['ego']['vehicle_state'], opp_state=info['oppo']['vehicle_state'])
    opponent_controller.reset(vehicle_state=info['oppo']['vehicle_state'], opp_state=info['ego']['vehicle_state'])
    # kinBicycleMPC.reset(vehicle_state=info['oppo']['vehicle_state'])
    log_def = []
    log_att = []

    # Initialize variables
    rew, terminated, truncated = None, False, False
    episode_count = 0
    success_count = 0

    # Main simulation loop
    while True:
        # Get actions from both controllers
        # Note: Your step function can take anything that the environment outputs, including the entire info dictionary and the observation vector.
        # See the details in multibarc_env.py.
        ego_action, att_sol, opp_sol= ego_controller.step(vehicle_state=info['ego']['vehicle_state'], opp_state=info['oppo']['vehicle_state'], terminated=info['ego']['terminated'],
                                            lap_no=info['ego']['lap_no'])
        oppo_action, def_sol = opponent_controller.step(vehicle_state=info['oppo']['vehicle_state'], opp_state=info['ego']['vehicle_state'], att_sol=att_sol, terminated=info['oppo']['terminated'],
                                            lap_no=info['oppo']['lap_no'])
        # oppo_action, _ = kinBicycleMPC.step(vehicle_state=info['oppo']['vehicle_state'], terminated=info['ego']['terminated'],
        #                                     lap_no=info['ego']['lap_no'])
        # Step the environment
        ob, rew, terminated, truncated, info = env.step({'ego': ego_action, 'oppo': oppo_action})

        # Log the data.
        log_att.append(ego_controller.get_log_dict(att_sol, opp_sol=opp_sol))
        log_def.append(opponent_controller.get_log_dict(def_sol, opp_sol=att_sol))

        # Log episode results
        if terminated['__all__'] or truncated['__all__']:
            overtaking_status = 2 if att_sol["X"][0, 0] > def_sol["X"][0, 0] else -1
            log_data.append(
                {"def": log_def, "att": log_att, "overtaking_status": overtaking_status}
            )
            log_def = []
            log_att = []
            episode_count += 1
            ob, info = env.reset()
            ego_controller.reset(vehicle_state=info['ego']['vehicle_state'], opp_state=info['oppo']['vehicle_state'])
            opponent_controller.reset(vehicle_state=info['oppo']['vehicle_state'], opp_state=info['ego']['vehicle_state'])

            # Add a small delay between episodes
            time.sleep(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    params = vars(parser.parse_args())

    main(**params)
