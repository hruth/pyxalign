from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.transform import RotationOptions, ShearOptions
from pyxalign.autorunner.config import InitializationConfig
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.utils import convert_projection_dict_to_array


def build_complex_projections(
    standard_data: StandardData,
    config: InitializationConfig,
) -> ComplexProjections:
    """Build a ComplexProjections object from a StandardData and an InitializationConfig."""
    new_array_size = standard_data.get_minimum_size_for_projection_array()
    new_array_size += config.pad
    projection_array = convert_projection_dict_to_array(
        standard_data.projections, new_array_size, pad_with_mode=True
    )

    projection_options = ProjectionOptions()
    projection_options.experiment.laminography_angle = config.laminography_angle
    projection_options.experiment.pixel_size = standard_data.pixel_size
    if config.rotation_angle != 0:
        projection_options.input_processing.rotation = RotationOptions(
            enabled=True, angle=config.rotation_angle
        )
    if config.shear_angle != 0:
        projection_options.input_processing.shear = ShearOptions(
            enabled=True, angle=config.shear_angle
        )

    complex_projections = ComplexProjections(
        projections=projection_array,
        angles=standard_data.angles,
        scan_numbers=standard_data.scan_numbers,
        options=projection_options,
        probe_positions=list(standard_data.probe_positions.values()),
        probe=standard_data.probe,
        skip_pre_processing=False,
        file_paths=list(standard_data.file_paths.values()),
    )
    if config.remove_scan_numbers is not None:
        complex_projections.drop_projections(config.remove_scan_numbers)
    return complex_projections