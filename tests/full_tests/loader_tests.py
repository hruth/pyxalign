# this file currently only tests the pear v3 loader
import os

from pyxalign import options as opts
from pyxalign.io.loaders.xrf.api import convert_xrf_projection_dicts_to_arrays
from pyxalign.test_utils_2 import CITestArgumentParser, CITestHelper

import data_loaders
from conftest import register_processing_function

ci_filename_prefix = "load_test"


@register_processing_function("cSAXS_e18044_LamNI_201907_loading_test")
def load_cSAXS_e18044_LamNI_201907_test_func(
    update_tester_results: bool = False, save_temp_files: bool = False
) -> dict[str, bool]:
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


@register_processing_function("2ide_ptycho_loading_test")
def load_2ide_ptycho_test_func(
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


@register_processing_function("2ide_xrf_loading_test")
def load_2ide_xrf_test_func(
    update_tester_results: bool = False, save_temp_files: bool = False
) -> dict[str, bool]:
    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name=os.path.join("2ide", "2025-1_Lamni-4"),
        update_tester_results=update_tester_results,
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)

    xrf_standard_data_dict, extra_PVs = data_loaders.load_2ide_xrf_test_data()

    # save/compare CI results
    for channel, standard_data in xrf_standard_data_dict.items():
        ci_test_helper.save_or_compare_results(
            standard_data.angles, f"{ci_filename_prefix}_" + f"standard_data_angles_{channel}"
        )
        ci_test_helper.save_or_compare_results(
            standard_data.scan_numbers,
            f"{ci_filename_prefix}_" + f"standard_data_scan_numbers_{channel}",
        )
        scan10 = standard_data.scan_numbers[10]
        ci_test_helper.save_or_compare_results(
            standard_data.projections[scan10],
            f"{ci_filename_prefix}_" + f"standard_data_projections10_{channel}",
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

    ci_test_helper.finish_test()
    return ci_test_helper.test_result_dict


def test_single_result(test_name, result):
    """
    The conftest.pytest_generate_tests hook uses this to parameterize the
    results of the registered processing functions
    """
    assert result, f"Check '{test_name}' failed"


if __name__ == "__main__":
    ci_parser = CITestArgumentParser()
    args = ci_parser.parser.parse_args()
    load_cSAXS_e18044_LamNI_201907_test_func(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
    )
    load_2ide_ptycho_test_func(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
    )
    load_2ide_xrf_test_func(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
    )
