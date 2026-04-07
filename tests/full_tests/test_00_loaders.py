# this file currently only tests the pear v3 loader
import os

import pytest

from pyxalign import options as opts
from pyxalign.io.loaders.xrf.api import convert_xrf_projection_dicts_to_arrays
from pyxalign.test_utils_2 import CITestArgumentParser, CITestHelper, primary_ci_test_folder_string

import data_loaders
from pyxalign.test_utils_2 import skip_if_data_not_found

ci_filename_prefix = "load_test"


def generate_results_load_cSAXS_e18044_LamNI_201907(
    update_tester_results: bool = False, save_temp_files: bool = False
) -> dict:
    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name="cSAXS_e18044_LamNI_201907",
        update_tester_results=update_tester_results,
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)

    # load data
    n_scans = 50
    lamni_data = data_loaders.load_cSAXS_e18044_LamNI_201907_test_data(
        scan_start=2714, scan_end=2714 + n_scans
    )

    selected_scan = lamni_data.scan_numbers[10]

    # save/compare results
    ci_test_helper.save_or_compare_results(
        lamni_data.projections[selected_scan], f"{ci_filename_prefix}_" + "selected_projection"
    )
    ci_test_helper.save_or_compare_results(
        lamni_data.probe, f"{ci_filename_prefix}_" + "lamni_data_probe"
    )
    ci_test_helper.save_or_compare_results(
        lamni_data.scan_numbers, f"{ci_filename_prefix}_" + "lamni_data_scan_numbers"
    )
    ci_test_helper.save_or_compare_results(
        lamni_data.angles, f"{ci_filename_prefix}_" + "lamni_data_angles"
    )
    ci_test_helper.save_or_compare_results(
        lamni_data.probe_positions[2730], f"{ci_filename_prefix}_" + "selected_probe_positions"
    )

    ci_test_helper.finish_test()
    return ci_test_helper.test_result_dict


@pytest.fixture(scope="module")
def results_cSAXS():
    return generate_results_load_cSAXS_e18044_LamNI_201907()


cSAXS_test_names = [
    f"{ci_filename_prefix}_" + x
    for x in [
        "selected_projection",
        "lamni_data_probe",
        "lamni_data_scan_numbers",
        "lamni_data_angles",
        "selected_probe_positions",
    ]
]


sub_path = "cSAXS_e18044_LamNI_201907"
reason = f"Expected data folder does not exist: {os.path.join(os.environ[primary_ci_test_folder_string], sub_path)}"


@pytest.mark.skipif(skip_if_data_not_found(sub_path), reason=reason)
@pytest.mark.parametrize("key", cSAXS_test_names)
def test_cSAXS_loading(results_cSAXS, key):
    assert results_cSAXS[key]


#### 2IDE ptycho data test ####


def generate_results_load_2ide_ptycho(
    update_tester_results: bool = False, save_temp_files: bool = False
) -> dict[str, bool]:
    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name=os.path.join("2ide", "2025-1_Lamni-6"),
        update_tester_results=update_tester_results,
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)

    # load data
    standard_data = data_loaders.load_2ide_ptycho_test_data()

    # save/compare CI results
    scan_10 = standard_data.scan_numbers[10]
    ci_test_helper.save_or_compare_results(
        standard_data.probe, f"{ci_filename_prefix}_" + "standard_data_probe"
    )
    ci_test_helper.save_or_compare_results(
        standard_data.scan_numbers, f"{ci_filename_prefix}_" + "standard_data_scan_numbers"
    )
    ci_test_helper.save_or_compare_results(
        standard_data.angles, f"{ci_filename_prefix}_" + "standard_data_angles"
    )
    ci_test_helper.save_or_compare_results(
        standard_data.probe_positions[scan_10], f"{ci_filename_prefix}_" + "probe_positions_10"
    )

    ci_test_helper.finish_test()
    return ci_test_helper.test_result_dict


@pytest.fixture(scope="module")
def results_2IDE_ptycho():
    return generate_results_load_2ide_ptycho()


test_names_2ide_ptycho = [
    f"{ci_filename_prefix}_" + x
    for x in [
        "standard_data_probe",
        "standard_data_scan_numbers",
        "standard_data_angles",
        "probe_positions_10",
    ]
]

sub_path = os.path.join("2ide", "2025-1_Lamni-6")
reason = f"Expected data folder does not exist: {os.path.join(os.environ[primary_ci_test_folder_string], sub_path)}"


@pytest.mark.skipif(skip_if_data_not_found(sub_path), reason=reason)
@pytest.mark.parametrize("key", test_names_2ide_ptycho)
def test_2ide_ptycho_loading(results_2IDE_ptycho, key):
    assert results_2IDE_ptycho[key]


### 2IDE XRF Loading Test ###

selected_xrf_channel = "Ti"


def generate_results_load_2ide_xrf(
    update_tester_results: bool = False, save_temp_files: bool = False
) -> dict[str, bool]:
    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name=os.path.join("2ide", "2025-1_Lamni-4"),
        update_tester_results=update_tester_results,
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)

    xrf_standard_data_dict = data_loaders.load_2ide_xrf_test_data()

    # save/compare CI results
    # selected_channel = "Ti"
    standard_data = xrf_standard_data_dict[selected_xrf_channel]
    ci_test_helper.save_or_compare_results(
        standard_data.angles,
        f"{ci_filename_prefix}_" + f"standard_data_angles_{selected_xrf_channel}",
    )
    ci_test_helper.save_or_compare_results(
        standard_data.scan_numbers,
        f"{ci_filename_prefix}_" + f"standard_data_scan_numbers_{selected_xrf_channel}",
    )
    scan10 = standard_data.scan_numbers[10]
    ci_test_helper.save_or_compare_results(
        standard_data.projections[scan10],
        f"{ci_filename_prefix}_" + f"standard_data_projections10_{selected_xrf_channel}",
    )

    # put data into dict of arrays
    xrf_array_dict = convert_xrf_projection_dicts_to_arrays(
        xrf_standard_data_dict,
        pad_with_mode=True,
    )

    for channel, projection_array in xrf_array_dict.items():
        ci_test_helper.save_or_compare_results(
            projection_array[:3], f"{ci_filename_prefix}_" + f"input_projections_{channel}"
        )
    print(standard_data.scan_numbers)
    ci_test_helper.finish_test()
    return ci_test_helper.test_result_dict


@pytest.fixture(scope="module")
def results_2IDE_xrf():
    return generate_results_load_2ide_xrf()


test_names_2ide_xrf = [
    f"{ci_filename_prefix}_" + x
    for x in [
        f"standard_data_angles_{selected_xrf_channel}",
        f"standard_data_scan_numbers_{selected_xrf_channel}",
        f"standard_data_projections10_{selected_xrf_channel}",
    ]
]

sub_path = os.path.join("2ide", "2025-1_Lamni-4")
reason = f"Expected data folder does not exist: {os.path.join(os.environ[primary_ci_test_folder_string], sub_path)}"


@pytest.mark.skipif(skip_if_data_not_found(sub_path), reason=reason)
@pytest.mark.parametrize("key", test_names_2ide_xrf)
def test_2ide_xrf_loading(results_2IDE_xrf, key):
    assert results_2IDE_xrf[key]


if __name__ == "__main__":
    ci_parser = CITestArgumentParser()
    args = ci_parser.parser.parse_args()
    # generate_results_load_cSAXS_e18044_LamNI_201907(
    #     update_tester_results=args.update_results,
    #     save_temp_files=args.save_temp_results,
    # )
    generate_results_load_2ide_ptycho(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
    )
    # generate_results_load_2ide_xrf(
    #     update_tester_results=args.update_results,
    #     save_temp_files=args.save_temp_results,
    # )