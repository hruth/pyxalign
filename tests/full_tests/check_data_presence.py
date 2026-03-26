import os
from pyxalign.test_utils_2 import primary_ci_test_folder_string


def skip_if_data_not_found(test_data_sub_path: str) -> bool:
    # test_data_names = [
    #     "cSAXS_e18044_LamNI_201907",
    #     os.path.join("2ide", "2025-1_Lamni-6"),
    #     os.path.join("2ide", "2025-1_Lamni-4"),
    # ]
    # for sub_path in test_data_names:
    data_path = os.path.join(os.environ[primary_ci_test_folder_string], test_data_sub_path)
    return not os.path.exists(data_path)

