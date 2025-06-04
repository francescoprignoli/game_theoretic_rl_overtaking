import numpy as np
import argparse
from pathlib import Path
import yaml
from matplotlib import pyplot as plt
from mpclab_common.track import get_track


def main(file, root=None):
    if root is None:
        root = Path.home() / 'Documents' / 'francesco-racing' / 'data'
    file_path = root / file
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
    ax.plot(data['attacker_state'][:, 0], data['attacker_state'][:, 1])
    track_obj.plot_map(ax=ax)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str)
    parser.add_argument('-f', '--file', type=str, required=True)
    args = parser.parse_args()
    main(file=args.file, root=args.root)
