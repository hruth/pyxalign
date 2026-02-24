import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pyxalign.autorunner.autorunner import AutorunnerPtychoV2


def run(config_file_path=None):
    autorunner = AutorunnerPtychoV2(config_file_path)
    autorunner.run()


if __name__ == "__main__":
    autorunner = run(
        config_file_path="/local/ci_tests/pyxalign/cSAXS_e18044_LamNI_201907/autorunner/autorunner_config.yaml"
    )
