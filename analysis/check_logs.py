import numpy as np
import argparse
from pathlib import Path
import yaml
from matplotlib import pyplot as plt
from mpclab_common.track import get_track

from mpcexp.utils.utils_fun import save_sim_data, dat2pkl, load_raw_data
from collections import defaultdict


def main(dir_name, root=None, player=None):
    data, size = load_raw_data(dir_name, root, player)
    track_obj = get_track('L_track_barc')
    fig, ax = plt.subplots()
    ax.plot(data['attacker_state'][:, 0], data['attacker_state'][:, 1], '.-', label='attacker')
    ax.plot(data['defender_state'][:, 0], data['defender_state'][:, 1], '.-', label='defender')
    track_obj.plot_map(ax=ax)
    ax.set_aspect('equal')
    ax.legend()
    # plt.show()

    fig, ax = plt.subplots()
    ax.hist(np.diff(data['timestamp']))
    ax.set_title('Solve time ')
    plt.show()

    # print(f"Loaded data for {len(data)} variables")
    # for key, value in data.items():
    #     print(f"{key}: {value.shape}")
    dat2pkl(data, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str)
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-p', '--player', type=str)
    args = parser.parse_args()
    main(dir_name=args.file, root=args.root, player=args.player)
