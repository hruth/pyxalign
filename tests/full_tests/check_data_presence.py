import os
from pyxalign.test_utils_2 import primary_ci_test_folder_string


def skip_if_data_not_found(test_data_sub_path: str) -> bool:
    data_path = os.path.join(os.environ[primary_ci_test_folder_string], test_data_sub_path)
    return not os.path.exists(data_path)

