"""tests/test_types.py — Type system unit tests"""
import ast
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from gpucc.types import (
    Int32Type, Int64Type, Float32Type, BoolType, PointerType, ArrayType,
    f32, i32, i64, bool_t, pointer_to, parse_annotation,
)


def test_scalar_ptx_props():
    assert i32.ptx_reg_prefix() == "%r"
    assert i32.ptx_reg_type() == ".u32"
    assert i32.byte_size() == 4

    assert f32.ptx_reg_prefix() == "%f"
    assert f32.ptx_reg_type() == ".f32"
    assert f32.byte_size() == 4

    assert i64.ptx_reg_prefix() == "%rd"
    assert i64.byte_size() == 8

    assert bool_t.ptx_reg_prefix() == "%p"
    assert bool_t.ptx_reg_type() == ".pred"


def test_pointer_type():
    p = pointer_to(f32)
    assert isinstance(p, PointerType)
    assert p.element_type == f32
    assert p.ptx_reg_prefix() == "%rd"
    assert p.byte_size() == 8


def test_array_type():
    arr = ArrayType(f32, (None, None))
    assert arr.ndim() == 2
    assert arr.element_type == f32


def test_parse_annotation_scalar():
    node = ast.parse("int32", mode="eval").body
    result = parse_annotation(node)
    assert result == i32

    node = ast.parse("float32", mode="eval").body
    result = parse_annotation(node)
    assert result == f32


def test_parse_annotation_array_1d():
    node = ast.parse("float32[N]", mode="eval").body
    result = parse_annotation(node)
    assert isinstance(result, ArrayType)
    assert result.element_type == f32
    assert result.shape == (None,)


def test_parse_annotation_array_2d():
    node = ast.parse("float32[M, K]", mode="eval").body
    result = parse_annotation(node)
    assert isinstance(result, ArrayType)
    assert result.ndim() == 2
    assert result.shape == (None, None)


def test_parse_annotation_array_fixed():
    node = ast.parse("int32[128]", mode="eval").body
    result = parse_annotation(node)
    assert isinstance(result, ArrayType)
    assert result.shape == (128,)


def test_parse_annotation_unknown_raises():
    node = ast.parse("complex128", mode="eval").body
    with pytest.raises(TypeError, match="Unknown type annotation"):
        parse_annotation(node)


def test_type_predicates():
    assert f32.is_float()
    assert not f32.is_int()
    assert i32.is_int()
    assert not i32.is_float()
    assert bool_t.is_bool()
    assert pointer_to(f32).is_pointer()
