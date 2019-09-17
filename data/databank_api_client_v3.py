#!/usr/bin/env python
#
# GNU General Public License https://www.gnu.org/licenses/gpl-3.0.txt
#
"""
Illinois Data Bank API client version 3
usage: databank_api_client_v3.py [-h] --dataset-key DATASET_KEY
                                 [--folder [FOLDER [FOLDER ...]]]
                                 [--file [FILE [FILE ...]]] [--as-subfolder]
                                 [--exclude-folders [EXCLUDE_FOLDERS [EXCLUDE_FOLDERS ...]]]
                                 [--temp-dir TEMP_DIR] --token TOKEN
                                 [--system {local,development,production,aws_test}]

Upload folder as zip file to Illinois Data Bank.

optional arguments:
  -h, --help            show this help message and exit
  --dataset-key DATASET_KEY
                        ID of the dataset in the IDB.
  --folder [FOLDER [FOLDER ...]]
                        Path to the folder.
  --file [FILE [FILE ...]]
                        Path to the files.
  --as-subfolder        Get sub folders from each folder as a single file.
  --exclude-folders [EXCLUDE_FOLDERS [EXCLUDE_FOLDERS ...]]
                        Exclude uploading these folders.
  --temp-dir TEMP_DIR   Folder where the zip files will be saved before
                        uploading.
  --token TOKEN         Token provided by IDB for file upload.
  --system {local,development,production,aws_test}
                        SYSTEM argument must be one of
                        local|development|production|pilot|aws_test,
                        production is default.

Example: 

python databank_api_client_v3.py --dataset-key <DATA_KEY> --token <TOKEN> --folder models --as-subfolder --exclude-folder models\all_multitask_shared models\all_multitask_shared_l2_0_lr_0.001

"""
from __future__ import division, print_function

import argparse
import logging
import os
import re
import sys
import tempfile
from shutil import make_archive

import requests
from tqdm import tqdm
from tusclient import client

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
    filename="databank_upload.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

# for multipart uploads, s3 requires that all chunks except last be at least 5 MB
FIVE_MB = 5 * 1024 * 1024  # 5MB

# If a SYSTEM argument is provided, validate it, otherwise use production as default.
VALID_SYSTEM_LIST = ["local", "development", "production", "aws_test"]

TUS_CLIENT_UPLOAD_REGEX = re.compile(r"^(?P<offset>[0-9]+) bytes uploaded \.\.\.$")


def get_filename(filepath):
    if os.path.isfile(filepath):
        file_info = os.stat(filepath)
        size = file_info.st_size
        filename = os.path.basename(filepath)
        print("uploading " + filename + " ...")
    else:
        raise RuntimeError(
            f"{filepath} must be the path to the file on the local filesystem to be uploaded.\n"
        )
    return filename, size


def generate_endpoints(dataset_key, system):
    """Generate Endpoints"""
    create_endpoint = "http"
    upload_endpoint = "http"
    if system == "production":
        create_endpoint += (
            "s://databank.illinois.edu/api/dataset/" + dataset_key + "/datafile"
        )
        upload_endpoint += "s://databank.illinois.edu/files/"

    elif system == "development":
        create_endpoint += (
            "s://rds-dev.library.illinois.edu/api/dataset/" + dataset_key + "/datafile"
        )
        upload_endpoint += "s://rds-dev.library.illinois.edu/files/"

    elif system == "local":
        create_endpoint += "://localhost:3000/api/dataset/" + dataset_key + "/datafile"
        upload_endpoint += "://localhost:3000/files/"

    elif system == "aws_test":
        create_endpoint += (
            "s://aws-databank-alb.library.illinois.edu/api/dataset/"
            + dataset_key
            + "/datafile"
        )
        upload_endpoint += "s://aws-databank-alb.library.illinois.edu/files/"

    else:
        raise RuntimeError(
            "Internal Error, please contact the Research Data Service databank@library.illinois.edu"
        )
    return create_endpoint, upload_endpoint


class IDBUploader(object):
    """Illinois data bank (IDB) uploader object.
    
    dataset_key: Dataset key provided by IDB
    token: Token provided by IDB
    system: One among the many valid systems.
    """

    def __init__(self, dataset_key, token, system="production", logger=None):
        self.dataset_key = dataset_key
        self.token = token
        self.create_endpoint, self.upload_endpoint = generate_endpoints(
            dataset_key, system
        )
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_client()

    def _setup_client(self):
        self.logger.info(f"Creating TusClient with token={self.token}")
        self.tus_client = client.TusClient(
            self.upload_endpoint, headers={"Authorization": f"Token token={self.token}"}
        )

    def upload_datafile(self, filepath):
        tus_client = self.tus_client
        token = self.token
        create_endpoint = self.create_endpoint
        dataset_key = self.dataset_key
        filename, size = get_filename(filepath)
        with tqdm(desc=f"Upload {filename}", unit="bytes", total=size) as pbar:

            def _log_func(msg):
                matches = TUS_CLIENT_UPLOAD_REGEX.match(msg)
                if matches:
                    offset = int(matches.group("offset"))
                    increment = offset - pbar.n
                    pbar.update(increment)
                    pbar.refresh()
                else:
                    pbar.set_postfix_str(msg)

            # set up tus client uploader
            uploader = tus_client.uploader(
                filepath, chunk_size=FIVE_MB, log_func=_log_func
            )
            # upload the entire file, chunk by chunk
            uploader.upload()
        # get the tus_url from the tus client uploader
        tus_url = uploader.url
        self.logger.info(f"Uploading {filepath} with size={size} as {filename}")
        headers = {"Authorization": f"Token token={token}"}
        data = {
            "filename": filename,
            "tus_url": tus_url,
            "size": size,
            "dataset_key": dataset_key,
        }
        create_response = requests.post(
            create_endpoint, headers=headers, data=data, verify=True
        )
        self.logger.info(f"Uploaded {filepath} to {create_response.text}")
        return create_response.text

    def upload_folder(self, folder, temp_dir):
        zip_filepath = folder_to_zipfile(folder, temp_dir)
        response_text = self.upload_datafile(zip_filepath)
        # Delete the file in temporary folder, else the disk space will blow up.
        if os.path.exists(zip_filepath):
            try:
                os.remove(zip_filepath)
                self.logger.info(f"Deleting {zip_filepath}.")
            except OSError as e:
                self.logger.error(f"{e.filename} - {e.strerror}.")
        else:
            self.logger.error(f"Cannot find {zip_filepath}.")
        return response_text


def create_parser():
    parser = argparse.ArgumentParser(
        description="Upload folder as zip file to Illinois Data Bank."
    )
    parser.add_argument(
        "--dataset-key", type=str, required=True, help="ID of the dataset in the IDB."
    )
    parser.add_argument(
        "--folder", nargs="*", type=str, default=(), help="Path to the folder."
    )
    parser.add_argument(
        "--file", nargs="*", type=str, default=(), help="Path to the files."
    )
    parser.add_argument(
        "--as-subfolder",
        action="store_true",
        help="Get sub folders from each folder as a single file.",
    )
    parser.add_argument(
        "--exclude-folders",
        type=str,
        default=(),
        nargs="*",
        help="Exclude uploading these folders.",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        help="Folder where the zip files will be saved before uploading.",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token provided by IDB for file upload.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="production",
        choices=VALID_SYSTEM_LIST,
        help="SYSTEM argument must be one of local|development|production|pilot|aws_test, production is default.",
    )
    return parser


class TqdmHandler(logging.StreamHandler):
    def __init__(self, pbar):
        logging.StreamHandler.__init__(self)
        self.pbar = pbar

    def emit(self, record):
        msg = self.format(record)
        self.pbar.set_postfix_str(msg)
        self.pbar.update(1)
        self.pbar.refresh()


def folder_to_zipfile(folder, temp_dir):
    base_dir = os.path.basename(folder)
    base_name = os.path.join(temp_dir, base_dir)
    root_dir = os.path.dirname(folder)
    logger = logging.getLogger(f"{__name__}.folder_to_zipfile")
    logger.info(f"Creating archive from {folder} and saving in {base_name}")
    with tqdm(desc=f"Archive {base_dir}") as pbar:
        handler = TqdmHandler(pbar)
        logger.addHandler(handler)
        filepath = make_archive(
            base_name, format="zip", root_dir=root_dir, base_dir=base_dir, logger=logger
        )
    return filepath


def main(args):
    logger.info(f"Arguments: {args}")
    temp_dir = args.temp_dir
    logger.info("Creating temporary directory.")
    temp_dir_obj = tempfile.TemporaryDirectory(dir=temp_dir)
    temp_dir = temp_dir_obj.name
    logger.info(f"Using temp directory: {temp_dir}")

    folders = args.folder
    exclude_folders = set(args.exclude_folders)
    filepaths = args.file if args.file is not None else []
    idb_uploader_obj = IDBUploader(
        dataset_key=args.dataset_key, token=args.token, system=args.system
    )
    try:
        for base_folder in folders:
            sub_folders = [base_folder]
            if args.as_subfolder:
                sub_folders = [
                    os.path.join(base_folder, dirname)
                    for dirname in os.listdir(base_folder)
                ]
                logger.info(f"Found {len(sub_folders)} in {base_folder}: {sub_folders}")
            for folder in sub_folders:
                if folder in exclude_folders:
                    logger.info(f"Excluding folder: {folder}")
                    continue
                logger.info(f"Uploading folder: {folder}")
                idb_uploader_obj.upload_folder(folder, temp_dir)
    finally:
        logger.info(f"Cleaning up temporary directory at: {temp_dir}")
        temp_dir_obj.cleanup()
    for filepath in filepaths:
        logger.info(f"Uploading filepath: {filepath}")
        idb_uploader_obj.upload_datafile(filepath)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
