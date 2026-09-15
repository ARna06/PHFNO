import inspect
import json
from pathlib import Path

import pytest

from experiments import vorticity_comparison
from experiments.comparison import ComparisonConfig


def test_training_cell_refreshes_a_stale_comparison_function(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "ipynb" / "compare_vorticity.ipynb"
    notebook = json.loads(path.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "vorticity_results =" in "".join(cell["source"]))

    def stale_comparison(datasets, config, output_dir):
        raise AssertionError("The notebook used the stale function")

    namespace = {
        "datasets": {},
        "config": ComparisonConfig(device="cpu", progress=False),
        "output": tmp_path,
        "velocity_results": {"config": {"seeds": [999]}},
        "run_vorticity_comparison": stale_comparison,
    }
    for _ in range(2):
        monkeypatch.setattr(vorticity_comparison, "run_vorticity_comparison", stale_comparison)
        with pytest.raises(ValueError, match="The velocity reference uses different seeds"):
            exec(source, namespace)
        function = vorticity_comparison.run_vorticity_comparison
        assert function is not stale_comparison
        assert "velocity_reference" in inspect.signature(function).parameters
