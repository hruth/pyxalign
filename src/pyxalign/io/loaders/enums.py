from enum import StrEnum, auto


class ExperimentInfoSourceType(StrEnum):
    LAMNI_DAT_FILE = auto()
    PTYCHO_FOLDERS = auto()
    BEAMLINE_2IDE_MDA_FILE = auto()

class ExperimentType(StrEnum):
    LYNX_PEAR = "LYNX: PEAR ptychography files"
    BEAMLINE_2IDE_PTYCHO_PEAR = "2-ID-E: PEAR ptychography files"
    BEAMLINE_2IDD_PTYCHO_PEAR = "2-ID-D: PEAR ptychography files"
    BEAMLINE_12IDE_PTYCHO_PEAR = "12-ID-E: PEAR ptychography files"
    BEAMLINE_2IDE_XRF = "2-ID-E: XRF"
    # BEAMLINE_2IDE_PTYCHO = "2-ID-E: ptycho"
    # BEAMLINE_2IDD_PTYCHO = "2-ID-D: ptycho"
    # BEAMLINE_12IDE_PTYCHO = "12-ID-E: ptycho"

    @classmethod
    def _missing_(cls, value):
        # backwards compatibility: old state files use "LYNX: ptycho"
        if value == "LYNX: ptycho":
            return cls.LYNX_PEAR
        return None


class MDAFilePatterns(StrEnum):
    """Possible file patterns of the files in the mda folder.

    These patterns allow the loading functions to correctly extract the
    scan number.

    """

    XFM_MDA_H5 = r"2xfm_(\d+)\.mda.h5"
    XFM_MDA = r"2xfm_(\d+)\.mda"
    BNP_FLY_MDA = r"bnp_fly(\d+)\.mda"


class RotationAnglePVStrings(StrEnum):
    """Possible values of strings used for accessing the rotation angle
    from the extra PVs dict."""

    XFM_M60_VAL = "2xfm:m60.VAL"
    IDBTAU_SM_ST_ACTPOS = "9idbTAU:SM:ST:ActPos"


class LaminoAnglePVStrings(StrEnum):
    """Possible values of strings used for accessing the rotation angle
    from the extra PVs dict."""

    XFM_M12_VAL = "2xfm:m12.VAL"