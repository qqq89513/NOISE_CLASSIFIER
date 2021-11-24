# =============================================================================
    # File name: data_parser.py
    # Author: Ting-Wei Zhang
    # Date created: 05/07/2021
    # Date last modified: 06/29/2013
    # Python Version: 3.8.8 64-bit
# =============================================================================
"""
    The script is built for parsing the datasets and saving the info.

    To execute this script:
    $ python ./utils/data_parser.py
    
    1. parse_noise_dir:
        to parse the Audioset directory but doesn't read any file.
        Instead, it walks to the subfolders and gets file paths by
        classes and save these into a .json file.
"""

import os, sys
sys.path.insert(0, os.getcwd())
import json
import vggish_params as params
def parse_noise_dir(splits_noise, noise_paths):
    # The dict to store the info
    json_content = {}

    # To parse all the splits
    for i in range(len(splits_noise)):
        # Add top level key to the dict
        json_content[splits_noise[i]] = {}

        # The number of folders in the splits represents
        # the class to be predicted by the model.
        noise_class = os.listdir(noise_paths[i])
        
        for j in range(len(noise_class)):
            # Add the noise class to the second level as a new key
            json_content[splits_noise[i]][noise_class[j]] = {}
            # A temp list to store filenames
            temp_list = []
            # dirpath, dirnames, filenames
            for root, dirs, files in os.walk(f'{noise_paths[i]}/{noise_class[j]}'):
                for filename in files:
                    file_path = f'{root}/{filename}'.replace("\\", "/")
                    if ((filename.endswith('.wav') or filename.endswith('.WAV')) and
                        os.path.getsize(file_path) > 44):
                        temp_list.append(file_path)
            
            # A temp dict to store the info
            temp_dict = {
                "records": len(temp_list),
                "filenames": temp_list
            }

            # Update the class with corresponding contents
            json_content[splits_noise[i]][noise_class[j]].update(temp_dict)

    # Save these info to a json file
    with open(params.PATH_NOISE_LIST, 'w') as fp:
        json.dump(json_content, fp, indent=4, ensure_ascii=False)

def start_parsing():
    # The splits of the noise dataset
    splits_noise = ["train", "test"]
    # Noise splits paths
    noise_paths = [params.PATH_NOISE_ROOT + split for split in splits_noise]
    parse_noise_dir(splits_noise, noise_paths)


if __name__ == "__main__":
    start_parsing()
