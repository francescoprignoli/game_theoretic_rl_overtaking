import argparse
import numpy as np
import mpcexp
from pathlib import Path
from mpcexp.utils.utils_fun import load_sim_data, dat2pkl, load_raw_data, unflatten_to_dict_of_lists


def main(filename, replay_speed=1, save_video=False, root=None):

    # ----- Load Data -----
    att_data, att_size = load_raw_data(filename, root, player='attacker')
    att_unflat_data = unflatten_to_dict_of_lists(att_data, log_size=att_size)

    def_data, def_size = load_raw_data(filename, root, player='defender')
    def_unflat_data = unflatten_to_dict_of_lists(def_data, log_size=def_size)

    size = min(att_size, def_size)
    unflat_data = {k: v[:size] for k, v in att_unflat_data.items()}
    unflat_data.update({k: v[:size] for k, v in def_unflat_data.items() if k != 'attacker_state'})

    # Fix key names
    # unflat_data["def"] = unflat_data.pop("log_def")
    # unflat_data["att"] = unflat_data.pop("log_att")

    unflat_data = [unflat_data]

    # dat2pkl(data, root)
    # else:
    #     data = load_sim_data(filename=filename)

    Ts = 0.1  # Sampling time in seconds
    track_file = "L_track_barc_race.json"  # Track file name
    vehicle_type = "barc"  # Vehicle type
    animator = mpcexp.Animator(
        vehicle_type=vehicle_type, Ts=Ts, track_file=track_file
    )

    # ----- Plot/Animate the Results -----
    try:
        mpcexp.IBR.plot_time([el["IBR_time"] for log in unflat_data for el in log["att"]])
        mpcexp.IBR.plot_time([el["IBR_time"] for log in unflat_data for el in log["def"]])
    except TypeError:
        import pdb
        pdb.set_trace()

    animator.animation_side_by_side(
        sim_data=unflat_data,
        save_video=save_video,
        replay_speed=replay_speed,
        follow_ego=False,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", '-f',
        type=str,
        required=True,
        help="Name of the simulation data file (required)",
    )
    parser.add_argument(
        "--replay_speed",
        type=float,
        default=1.0,
        help="Speed of the replay (default: 1.0)",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Flag to save the animation as a video file (default: False)",
    )
    args = parser.parse_args()

    main(args.filename, args.replay_speed, args.save_video)
