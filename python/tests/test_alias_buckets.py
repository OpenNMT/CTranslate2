"""Aliasing must retain the original alphabetical and numerical semantics."""
import numpy as np
import pytest

from ctranslate2.specs import model_spec as m
from ctranslate2.specs.drama3_spec import Drama3Spec


def original_aliases(spec):
    variables = spec.variables(ordered=True)
    quantizable = {}
    for name, _ in variables:
        scope, attr = m._parent_scope(name)
        quantizable[name] = hasattr(m.index_spec(spec, scope), attr + "_scale")
    for name, value in reversed(variables):
        for other_name, other_value in variables:
            if name == other_name:
                break
            scope, attr = m._parent_scope(name)
            if (
                not value.is_scalar()
                and quantizable[name] == quantizable[other_name]
                and value.equal(other_value)
                and attr not in m.SKIP_CREATING_ALIAS
            ):
                setattr(m.index_spec(spec, scope), attr, other_name)
                break


class Fixture(m.LayerSpec):
    _alias_variables = Drama3Spec._alias_variables

    def __init__(self, values):
        for name, value in values.items():
            setattr(self, name, value)


def aliases(spec):
    return {
        name: value
        for name, value in spec.variables().items()
        if isinstance(value, str)
    }


def mixed_fixture():
    nan = np.array([np.nan, 3], np.float32)
    zero = np.array([-0.0, 1], np.float32)
    values = {
        "a_nan": nan,
        "b_nan_identity": nan,
        "c_nan_copy": nan.copy(),
        "d_zero": zero,
        "e_positive_zero": np.array([0.0, 1], np.float32),
        "f_first_collision": np.array([0.0, 2], np.float32),
        "g_scalar": np.float32(1),
        "h_scalar": np.float32(1),
        "i_transposed": np.arange(6, dtype=np.float32).reshape(2, 3).T,
        "j_contiguous": np.arange(6, dtype=np.float32).reshape(2, 3).T.copy(),
        "k_integer": np.array([0, 1], np.int32),
        "l_shape": np.array([[0.0, 1]], np.float32),
        "aa": Fixture({"rotary_scaling_long_factor": np.array([7.0, 8], np.float32)}),
        "zz": Fixture({"weight": np.array([7.0, 8], np.float32)}),
    }
    for name, quantized in [("q", True), ("r", False), ("s", True)]:
        layer = {"weight": np.array([[1.0001, -0.33331]], np.float32)}
        if quantized:
            layer["weight_scale"] = m.OPTIONAL
        values[name] = Fixture(layer)
    if m.torch_is_available:
        torch = m.torch
        for dtype in (torch.bfloat16, torch.float32):
            prefix = str(dtype).split(".")[-1]
            tensor = torch.tensor([float("nan"), 3], dtype=dtype)
            values[prefix] = Fixture(
                {
                    "a_nan": tensor,
                    "b_identity": tensor,
                    "c_copy": tensor.clone(),
                    "d_zero": torch.tensor([-0.0, 1], dtype=dtype),
                    "e_positive_zero": torch.tensor([0.0, 1], dtype=dtype),
                    "f_first_collision": torch.tensor([0.0, 2], dtype=dtype),
                    "g_scalar": torch.tensor(1.0, dtype=dtype),
                    "h_scalar": torch.tensor(1.0, dtype=dtype),
                    "i_transposed": torch.arange(6, dtype=dtype).reshape(2, 3).T,
                    "j_contiguous": torch.arange(6, dtype=dtype)
                    .reshape(2, 3)
                    .T.contiguous(),
                }
            )
    result = Fixture(values)
    result.validate()
    return result


def test_grouping_matches_original_numpy_and_torch_aliases():
    expected, actual = mixed_fixture(), mixed_fixture()
    original_aliases(expected)
    actual._alias_variables()
    assert aliases(actual) == aliases(expected)
    assert aliases(actual)["b_nan_identity"] == "a_nan"
    assert "c_nan_copy" not in aliases(actual)
    assert aliases(actual)["e_positive_zero"] == "d_zero"
    assert aliases(actual)["s/weight"] == "q/weight"
    assert "r/weight" not in aliases(actual)
    assert aliases(actual)["zz/weight"] == "aa/rotary_scaling_long_factor"


@pytest.mark.parametrize("same_object", [False, True])
def test_empty_arrays_keep_original_identity_and_error_behavior(same_object):
    def fixture():
        empty = np.empty((0,), np.float32)
        spec = Fixture({"a": empty, "b": empty if same_object else empty.copy()})
        spec.validate()
        return spec

    expected, actual = fixture(), fixture()
    if same_object:
        original_aliases(expected)
        actual._alias_variables()
        assert aliases(actual) == aliases(expected) == {"b": "a"}
    else:
        with pytest.raises(IndexError):
            original_aliases(expected)
        with pytest.raises(IndexError):
            actual._alias_variables()


def test_distinct_first_values_avoid_quadratic_exact_comparisons():
    class Counted(m.NumpyVariable):
        calls = 0

        def equal(self, other):
            type(self).calls += 1
            return super().equal(other)

    def fixture():
        return Fixture(
            {
                f"v{i:04d}": Counted(np.array([0] * 7 + [i], np.int32))
                for i in range(1000)
            }
        )

    original_aliases(fixture())
    assert Counted.calls == 1000 * 999 // 2
    Counted.calls = 0
    fixture()._alias_variables()
    assert Counted.calls == 0


def test_matching_eight_value_prefix_still_checks_full_arrays():
    first = np.arange(12, dtype=np.float32)
    different = first.copy()
    different[-1] = 99
    values = {"a": first, "b": different, "c": different.copy()}
    expected, actual = Fixture(values), Fixture(values)
    expected.validate()
    actual.validate()
    original_aliases(expected)
    actual._alias_variables()
    assert aliases(actual) == aliases(expected) == {"c": "b"}


def test_other_models_keep_cross_quantization_aliases():
    spec = Fixture(
        {
            "a": Fixture({"weight": np.array([[1.0, 2.0]], np.float32)}),
            "b": Fixture(
                {
                    "weight": np.array([[1.0, 2.0]], np.float32),
                    "weight_scale": m.OPTIONAL,
                }
            ),
        }
    )
    spec.validate()
    m.LayerSpec._alias_variables(spec)
    assert aliases(spec) == {"b/weight": "a/weight"}
