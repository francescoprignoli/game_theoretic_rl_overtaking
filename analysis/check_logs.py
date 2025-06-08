import numpy as np
import argparse
from pathlib import Path
import yaml
from matplotlib import pyplot as plt
from mpclab_common.track import get_track

from mpcexp.utils.utils_fun import save_sim_data
from collections import defaultdict 


def main(dir_name, root=None):
    if root is None:
        root = Path.home() / 'Documents' / 'francesco-racing' / 'data'
    file_path = root / dir_name
    if not file_path.exists() or file_path.is_file():
        raise FileNotFoundError(file_path)
    size = np.memmap(file_path / 'metadata.dat', mode='r', dtype=np.uint16)[0]
    print(f"Found {size} entries in the dataset")
    with open(file_path / 'shapes.yaml', 'r') as f:
        shapes = yaml.load(f, Loader=yaml.FullLoader)

    data = {}
    for file in file_path.iterdir():
        if not file.is_file() or file.name == 'metadata.dat' or file.name == 'shapes.yaml':
            continue
        name = file.stem
        data[name] = np.array(np.memmap(file, mode='r', dtype=np.float64, shape=shapes[name]))[:size]

    track_obj = get_track('L_track_barc')
    fig, ax = plt.subplots()
    ax.plot(data['attacker_state'][:, 0], data['attacker_state'][:, 1], '.-')
    track_obj.plot_map(ax=ax)
    ax.set_aspect('equal')
    plt.show()

    # print(f"Loaded data for {len(data)} variables")
    # for key, value in data.items():
    #     print(f"{key}: {value.shape}")

    # Unflatten the log
    unflat_data = unflatten_to_dict_of_lists(data, log_size=size)
    # Fix key names
    unflat_data["def"] = unflat_data.pop("log_def")
    unflat_data["att"] = unflat_data.pop("log_att")

    # Save the unflattened data
    experiment = "RA-GTP"
    defender_form = "fast"
    vehicle_type = "barc"
    filename = f"barc_exp_{dir_name}_{experiment}_{defender_form}_{vehicle_type}"
    save_sim_data([unflat_data], format='.pkl', filename=filename)
    print(f"Saved data to {filename}.pkl")

    
def unflatten_to_dict_of_lists(flat_log, log_size, sep='-'):
    log_data = defaultdict(list)

    for t in range(log_size):
        for flat_key, array in flat_log.items():
            keys = flat_key.split(sep)
            root = keys[0]

            # ensure top-level dictionary
            while len(log_data[root]) < t + 1:
                log_data[root].append({}) if len(keys) > 1 else log_data[root].append(None)

            if len(keys) == 1:
                log_data[root][t] = array[t]
            else:
                curr = log_data[root][t]
                for k in keys[1:-1]:
                    curr = curr.setdefault(k, {})
                curr[keys[-1]] = array[t]

    return dict(log_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str)
    parser.add_argument('-f', '--file', type=str, required=True)
    args = parser.parse_args()
    main(dir_name=args.file, root=args.root)
