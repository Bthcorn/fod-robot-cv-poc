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
