import requests
from tqdm import tqdm
from typing import Tuple, List, Union, Dict, Callable, Any, Optional
import os
import json
import sys
from time import time
# name: the version of whoiswho data, ['v1', 'v2', 'v3']
# type: data partition of the data, ['train', 'valid', 'test']
# path: saved dir
# partition: none
DATA_PATH = "https://lfs.aminer.cn/misc/ND-data"
NAME_SET = set(['v1', 'v2', 'v3'])
TYPE_SET = set(['train', 'valid', 'test'])
TASK_SET = set(['SND','RND'])
# function to display download progress bar
def progress_bar(progress, total, speed):
    filled = int(progress * 40 // total)
    bar = '█' * filled + '-' * (40 - filled)
    sys.stdout.write('\r|%s| %d%% (%.2f KB/s)' % (bar, progress / total * 100, speed/1024))
    sys.stdout.flush()


def LoadData(name: str, type: str, task: str, path = None, just_version=False) -> List[dict]:
    if name not in NAME_SET:
        raise ValueError(f"NAME must in {NAME_SET}")
    if type not in TYPE_SET:
        raise ValueError(f"TYPE must in {TYPE_SET}")
    if task not in TASK_SET:
        raise ValueError(f"TASK must in {TASK_SET}")
    version={"name":name,"task":task,"type":type}
    if just_version:
        return version

    ret = []
    if path is None:
        path = os.path.dirname(__file__)
    url_list = []
    # Check if the download path exists, and create it if it doesn't
    download_data_path = os.path.join(path, f"data/{name}/{task}/{type}")
    if not os.path.exists(download_data_path):
        os.makedirs(download_data_path)

    # Define the URL of the data to download
    if type == 'train':
        url = os.path.join(DATA_PATH,
                        f"na-{name}",
                        f"train_author.json")
        url_list.append((url, "train_author.json"))
        url = os.path.join(DATA_PATH,
                        f"na-{name}",
                        f"train_pub.json")
        url_list.append((url, "train_pub.json"))
    elif type == 'valid':
        if task== 'RND':
            url = os.path.join(DATA_PATH,
                            f"na-{name}",
                            f"whole_author_profiles.json")
            url_list.append((url, "whole_author_profiles.json"))
            url = os.path.join(DATA_PATH,
                            f"na-{name}",
                            f"whole_author_profiles_pub.json")
            url_list.append((url, "whole_author_profiles_pub.json"))
            url = os.path.join(DATA_PATH,
                               f"na-{name}",
                               f"cna_valid_unass.json")
            url_list.append((url, "cna_valid_unass.json"))
            url = os.path.join(DATA_PATH,
                               f"na-{name}",
                               f"cna_valid_unass_pub.json")
            url_list.append((url, "cna_valid_unass_pub.json"))
            url = os.path.join(DATA_PATH,
                               f"na-{name}",
                               f"cna_valid_ground_truth.json")
            url_list.append((url, "cna_valid_ground_truth.json"))

        else:
            url = os.path.join(DATA_PATH,
                            f"na-{name}",
                            f"sna_valid_raw.json")
            url_list.append((url, "sna_valid_raw.json"))
            url = os.path.join(DATA_PATH,
                               f"na-{name}",
                               f"sna_valid_pub.json")
            url_list.append((url, "sna_valid_pub.json"))
            url = os.path.join(DATA_PATH,
                               f"na-{name}",
                               f"sna_valid_example.json")
            url_list.append((url, "sna_valid_example.json"))

    else :
        if task == 'RND':
            url = os.path.join(DATA_PATH,
                               f"na-{name}",
                               f"cna_test_unass.json")
            url_list.append((url, "cna_test_unass.json"))
            url = os.path.join(DATA_PATH,
                               f"na-{name}",
                               f"cna_test_unass_pub.json")
            url_list.append((url, "cna_test_unass_pub.json"))
        else:
            url = os.path.join(DATA_PATH,
                            f"na-{name}",
                            f"sna_test_raw.json")
            url_list.append((url, "sna_test_raw.json"))
            url = os.path.join(DATA_PATH,
                               f"na-{name}",
                               f"sna_test_pub.json")
            url_list.append((url, "sna_test_pub.json"))


    # Define the request headers
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}


    for url, filename in url_list:
        if os.path.exists(os.path.join(download_data_path,filename)):
            print(f"{filename} already downloaded...")
            with open(os.path.join(download_data_path, filename), 'r') as f:
                content = json.load(f)
            ret.append(content)
            continue
        # print(f"Downloading {filename}")
        # Send the request and get the response
        response = requests.get(url, headers=headers, stream=True)

        # check if the request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.reason}")
            response.raise_for_status()

        # Get the total size of the file in bytes
        total_size = int(response.headers.get('content-length', 0))

        # Define a chunk size of 1 MB
        chunk_size = 1024 * 1024

        # open a file for writing in JSON format
        progress = 0
        start_time = None
        speed = 0

        # open a file for writing in JSON format
        with open(os.path.join(download_data_path, filename), 'w') as f:

            # iterate over the response content in chunks
            for chunk in response.iter_content(chunk_size=1024):

                # calculate the progress and download speed
                progress += len(chunk)
                if not start_time:
                    start_time = time()
                else:
                    download_time = time() - start_time
                    speed = progress / download_time

                # write the chunk to the file
                f.write(chunk.decode('utf-8', errors='ignore'))

                # update the progress bar and download speed
                progress_bar(progress, total_size, speed)

        # check if the file was downloaded successfully
        if total_size != 0 and progress != total_size:
            print(f"\nError: failed to download the {filename} file")
        else:
            print(f"\nDownload {filename} successful!")

        with open(os.path.join(download_data_path, filename), 'r') as f:
            content = json.load(f)
        ret.append(content)
    
    return ret,version



if __name__ == '__main__':
    '''RND task'''
    # 0: train_author.json  1: train_pub.json
    # train,version = LoadData(name="v3", type="train",task='RND')

    # 0: whole_author_profiles.json 1: whole_author_profiles_pub.json  2: cna_valid_unass.json 3: cna_valid_unass_pub.json
    # 4: cna_valid_ground_truth.json
    # valid,version = LoadData(name="v3", type="valid",task='RND')

    # 0: cna_test_unass.json 1: cna_test_unass_pub.json
    # test,version = LoadData(name="v3", type="test", task='RND')

    '''SND task'''
    # 0: train_author.json  1: train_pub.json
    # train, version = LoadData(name="v3", type="train", task='SND')
    # 0: sna_valid_raw.json  1: sna_valid_pub.json  2: sna_valid_example.json
    # valid,version = LoadData(name="v3", type="valid",task='SND')
    # 0: cna_test_unass.json 1: cna_test_unass_pub.json
    # test,version = LoadData(name="v3", type="test", task='SND')



