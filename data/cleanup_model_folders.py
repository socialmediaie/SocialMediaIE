import argparse
import logging
import os
from glob import glob

from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
    filename="cleaup.log",
    filemode='w'
)
logger = logging.getLogger(__name__)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Cleanup model directories only retaining essential files."
    )
    parser.add_argument(
        "--base-folders",
        nargs="+",
        required=True,
        help="Folders containing the model files.",
    )
    parser.add_argument(
        "--as-subfolder",
        action="store_true",
        help="Get sub folders from each folder as base folder",
    )
    parser.add_argument(
        "--remove-best",
        action="store_true",
        help="Remove best.th",
    )
    parser.add_argument("--exclude-folders", nargs="*", help="Subfolders to exclude.")
    return parser


def get_subfolders(base_folder):
    sub_folders = [
        os.path.join(base_folder, dirname) for dirname in os.listdir(base_folder)
    ]
    return sub_folders


def cleanup_folder(folder, remove_best=False):
    model_states = glob(f"{folder}/model_state_epoch_*.th")
    training_states = glob(f"{folder}/training_state_epoch_*.th")
    all_filenames = model_states + training_states
    if remove_best:
        all_filenames.extend(glob(f"{folder}/best.th"))
    with tqdm(desc=f"{folder}", total=len(all_filenames)) as pbar:
        for filename in all_filenames:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except OSError as e:
                    pbar.write(f"{e.filename} - {e.strerror}.")
            else:
                pbar.write(f"Cannot find {filename}.")
            pbar.set_postfix_str(filename)
            pbar.update(1)
            pbar.refresh()


def main(args):
    base_folders = args.base_folders
    exclude_folders = set()
    if args.exclude_folders is not None:
        exclude_folders = set(exclude_folders)
    for base_folder in base_folders:
        logger.info(f"Base folder: {base_folder}")
        sub_folders = [base_folder]
        if args.as_subfolder:
            sub_folders = get_subfolders(base_folder)
        logger.info(f"Cleaning up {len(sub_folders)} folders: {sub_folders}")
        for folder in tqdm(sub_folders, desc="Sub folder"):
            if folder in exclude_folders:
                tqdm.write(f"Excluding {folder}")
                continue
            cleanup_folder(folder, remove_best=args.remove_best)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    main(args)
