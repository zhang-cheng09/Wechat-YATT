# coding=utf-8
"""老模型加载 weight exporter 时不得引入 qwen4_exp modeling。"""
import os
import pathlib
import subprocess
import sys
import unittest


_WEIGHT_EXPORTOR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "gpatch_v4/training_backend/fsdp2_backend/weight_exportor.py"
)
_SCRIPT = r"""
import runpy
import sys

namespace = runpy.run_path(sys.argv[1])
model_arch = namespace["MODEL_ARCH"]
get_weight_exportor = namespace["get_weight_exportor"]

modeling = "gpatch_v4.models.qwen4_exp.modeling_qwen4_exp"
pkg = "gpatch_v4.models.qwen4_exp"
weight_export = "gpatch_v4.models.qwen4_exp.weight_export"
assert get_weight_exportor(model_arch.DEEPSEEK_V4) is not None
assert get_weight_exportor("llama") is None
for name in (modeling, pkg, weight_export):
    assert name not in sys.modules, name
"""


class Qwen4ExpImportIsolationTest(unittest.TestCase):
    def test_old_arch_weight_exportor_import_skips_qwen4_exp_modeling(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        subprocess.check_call(
            [sys.executable, "-c", _SCRIPT, str(_WEIGHT_EXPORTOR)], env=env
        )
