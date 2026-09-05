"""The a16 class-head interceptor. See export.a16_classification_head."""

import pytest

from fodcv.research.export import cls_layers_to_a16

# What Ultralytics' export_hailo actually assembles for a detect head, trimmed.
SCRIPT = "\n".join([
    "normalization1 = normalization([0, 0, 0], [255, 255, 255])",
    "model_optimization_flavor(optimization_level=2)",
    "change_output_activation(conv54, sigmoid)",
    "change_output_activation(conv65, sigmoid)",
    "change_output_activation(conv80, sigmoid)",
    'nms_postprocess("nms_config.json", meta_arch=yolov8, engine=cpu)',
])


def test_appends_a16_for_every_class_conv():
    out = cls_layers_to_a16(SCRIPT)
    assert out.startswith(SCRIPT)  # the original script survives untouched
    assert out.splitlines()[-1] == (
        "quantization_param([conv54, conv65, conv80], precision_mode=a16_w16)")


def test_reg_convs_are_left_at_8_bit():
    # Only the sigmoid-activated convs are class convs; the regression convs
    # share the same output list and must not be widened.
    widened = cls_layers_to_a16(SCRIPT + "\nchange_output_activation(conv51, none)").splitlines()[-1]
    assert "conv51" not in widened


def test_refuses_a_script_with_no_class_convs():
    # Silently compiling an untouched a8 script while reporting an a16 build is
    # the failure this whole change exists to stop.
    with pytest.raises(AssertionError):
        cls_layers_to_a16("model_optimization_flavor(optimization_level=2)")


# --- the two rewrites added for the `n`-at-640 investigation --------------------

def nms_script(tmp_path):
    """SCRIPT with a real nms_config.json beside it -- all_outputs_to_a16 reads
    the regression convs out of the file, because the script never names them."""
    import json
    config = tmp_path / "nms_config.json"
    config.write_text(json.dumps({"bbox_decoders": [
        {"name": "bbox_decoder_8", "stride": 8, "reg_layer": "conv51", "cls_layer": "conv54"},
        {"name": "bbox_decoder_16", "stride": 16, "reg_layer": "conv62", "cls_layer": "conv65"},
        {"name": "bbox_decoder_32", "stride": 32, "reg_layer": "conv77", "cls_layer": "conv80"},
    ]}))
    return SCRIPT.replace('nms_postprocess("nms_config.json"', f'nms_postprocess("{config}"')


def test_a16_all_widens_regression_convs_too(tmp_path):
    from fodcv.research.export import all_outputs_to_a16

    widened = all_outputs_to_a16(nms_script(tmp_path)).splitlines()[-1]

    # The class convs alone are what --a16-cls does; this must add the box branch.
    for layer in ("conv51", "conv54", "conv62", "conv65", "conv77", "conv80"):
        assert layer in widened, f"{layer} missing from {widened}"
    assert widened.endswith("precision_mode=a16_w16)")


def test_a16_all_refuses_a_script_with_no_nms_config():
    """Without the nms_postprocess line there is nowhere to read reg layers from,
    and silently widening only the class convs would masquerade as --a16-all."""
    from fodcv.research.export import all_outputs_to_a16

    with pytest.raises(AssertionError, match="nms_postprocess"):
        all_outputs_to_a16("change_output_activation(conv54, sigmoid)")


def test_optimization_level_is_substituted_not_appended():
    """A second model_optimization_flavor line would be a silent conflict."""
    from fodcv.research.export import raise_optimization_level

    out = raise_optimization_level(SCRIPT, 4)

    assert "optimization_level=4" in out
    assert "optimization_level=2" not in out
    assert out.count("model_optimization_flavor") == 1


def test_optimization_level_refuses_a_script_it_cannot_find_the_flavor_in():
    from fodcv.research.export import raise_optimization_level

    with pytest.raises(AssertionError, match="rewrote 0"):
        raise_optimization_level("change_output_activation(conv54, sigmoid)", 4)
