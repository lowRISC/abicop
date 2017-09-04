import pytest
from rvcc import *

def test_first_class_array_arg():
    with pytest.raises(ValueError):
        RVMachine().call([Array(Int8, 3)])

def test_first_class_array_ret():
    with pytest.raises(ValueError):
        RVMachine().call([], Array(Int8, 2))

def test_invalid_xlen():
    with pytest.raises(ValueError):
       RVMachine(xlen=16)
    with pytest.raises(ValueError):
       RVMachine(xlen=33)
    with pytest.raises(ValueError):
       RVMachine(xlen=256)

def test_invalid_flen():
    with pytest.raises(ValueError):
       RVMachine(flen=16)
    with pytest.raises(ValueError):
       RVMachine(flen=33)
    with pytest.raises(ValueError):
       RVMachine(flen=256)

def get_arg_gprs(state):
    return [state.typestr_or_name(state.gprs[idx]) for idx in range(10, 18)]

def get_arg_fprs(state):
    return [state.typestr_or_name(state.fprs[idx]) for idx in range(10, 18)]

def get_stack_objects(state):
    return [state.typestr_or_name(obj) for obj in state.stack]

def test_no_args_void_return():
    m = RVMachine(xlen=32)
    state = m.call([])
    assert(get_arg_gprs(state)[0:1] == ["?"])

def test_many_args():
    # The stack should be used when arg registers are exhausted
    m = RVMachine(xlen=32)
    state = m.call([UInt8, Char, Char, Char, Char, Char, Char,
        Int8, Int8, Int8, UInt128])
    assert(get_stack_objects(state) == ["arg08", "Pad24", "arg09", "Pad24", "&arg10"])

def test_2xlen_rv32i():
    # 2xlen arguments are passed in GPRs, which need not be 'aligned' register 
    # pairs
    m = RVMachine(xlen=32)
    state = m.call([Int64, Int32, Double, Struct(Int8, Int32, Int8)])
    assert(get_arg_gprs(state)[0:8] == ["arg00[0:31]", "arg00[32:63]", 
        "arg01", "arg02[0:31]", "arg02[32:63]", "arg03[0:31]", "arg03[32:63]", "?"])

    # If only one arg GPR is available, the other half goes on the stack
    state = m.call([Int8, Int8, Int8, Int8, Int8, Int8, Int8,
        Double])
    assert(get_arg_gprs(state)[6:8] == ["arg06", "arg07[0:31]"])
    assert(len(state.stack) == 1)
    assert(state.typestr_or_name(state.stack[0]) == "arg07[32:63]")

    # 2xlen arguments must have their alignment maintained when passed on the
    # stack
    state = m.call([Int8, Int8, Int8, Int8, Int8, Int8, Int8,
        Int8, Int8, Double])
    assert(get_stack_objects(state) == ["arg08", "Pad24", "arg09"])

def test_gt_2xlen_rv32i():
    # scalars and aggregates > 2xlen are passed indirect
    m = RVMachine(xlen=32)
    state = m.call([Int128, LongDouble, Struct(Int64, Double)])
    assert(get_arg_gprs(state)[0:4] == ["&arg00", "&arg01", "&arg02", "?"])

def test_fp_scalars_rv32ifd():
    m = RVMachine(xlen=32, flen=64)
    # FPRs should be used as well as GPRs
    state = m.call([Float, Int64, Double, Int32])
    assert(get_arg_gprs(state)[0:4] == ["arg01[0:31]", "arg01[32:63]", 
        "arg03", "?"])
    assert(get_arg_fprs(state)[0:3] == ["arg00", "arg02", "?"])

    # Use GPRs when FPR arg registers are exhausted
    state = m.call([Float, Double, Float, Double, Float, Double, Float,
        Double, Char, Double, Float])
    assert(get_arg_gprs(state)[0:5] == ["arg08", "arg09[0:31]", 
        "arg09[32:63]", "arg10", "?"])

    # A float might end up split between stack and GPRs due to the FPRs being 
    # exhausted
    state = m.call([Float, Int64, Double, Int64, Float, Int64, Double, Char,
        Float, Double, Float, Double, Double])
    assert(get_arg_gprs(state)[6:8] == ["arg07", "arg12[0:31]"])
    assert(get_stack_objects(state) == ["arg12[32:63]"])

    # Greater than flen, pass according to integer calling convention
    state = m.call([LongDouble])
    assert(get_arg_gprs(state)[0:2] == ["&arg00", "?"])

def test_fp_int_aggregates_rv32ifd():
    m = RVMachine(xlen=32, flen=64)
    # Float+float
    state = m.call([Struct(Double, Float)])
    assert(get_arg_fprs(state)[0:3] == ["arg00[0:63]", "arg00[64:95]", "?"])

    # Float+int, int+float
    state = m.call([Struct(Double, Int16)])
    assert(get_arg_gprs(state)[0:2] == ["arg00[64:79]", "?"])
    assert(get_arg_fprs(state)[0:2] == ["arg00[0:63]", "?"])
    state = m.call([Struct(Int8, Double)])
    assert(get_arg_gprs(state)[0:2] == ["arg00[0:7]", "?"])
    assert(get_arg_fprs(state)[0:2] == ["arg00[64:127]", "?"])

    # The "int" field can't be a small aggregate
    state = m.call([Struct(Struct(Int8, Int8), Float)])
    assert(get_arg_gprs(state)[0:3] == ["arg00[0:31]", "arg00[32:63]", "?"])
    assert(get_arg_fprs(state)[0] == "?")

    # Use integer calling convention if the int is greater than xlen or the 
    # float greater than flen
    state = m.call([Struct(Int64, Float)])
    assert(get_arg_gprs(state)[0:2] == ["&arg00", "?"])
    assert(get_arg_fprs(state)[0] == "?")
    state = m.call([Struct(Int32, LongDouble)])
    assert(get_arg_gprs(state)[0:2] == ["&arg00", "?"])
    assert(get_arg_fprs(state)[0] == "?")

    # Check flattening
    equiv_args = [
        [Struct(Int32, Struct(Double))],
        [Struct(Array(Int32, 1), Struct(Double))],
        [Struct(Array(Int32, 1), Array(Struct(Double), 1))],
        [Struct(Struct(Int32), Struct(), Struct(Double))],
        [Struct(Int32, Struct(Array(Double, 1)))],
        ]
    for args in equiv_args:
        state = m.call(args)
        assert(get_arg_gprs(state)[0:2] == ["arg00[0:31]", "?"])
        assert(get_arg_fprs(state)[0:2] == ["arg00[64:127]", "?"])

def test_var_args_wrapper():
    # Test that VarArgs can't be misused
    m = RVMachine(xlen=32)
    with pytest.raises(InvalidVarArgs):
        RVMachine().call([], VarArgs(Int32))
    with pytest.raises(InvalidVarArgs):
        RVMachine().call([VarArgs(Int32), VarArgs(Int64)])
    with pytest.raises(InvalidVarArgs):
        RVMachine().call([VarArgs(Int32), Int64])

def test_var_args():
    m = RVMachine(xlen=32, flen=64)

    state = m.call([Int32, VarArgs(Int32, Struct(Int64, Double))])
    assert(get_arg_gprs(state)[0:4] == ["arg00", "varg00", "&varg01", "?"])

    # 2xlen aligned and sized varargs are passed in an aligned register pair
    state = m.call([Int32, VarArgs(Int64)])
    assert(get_arg_gprs(state)[0:4] == ["arg00", "?", "varg00[0:31]", "varg00[32:63]"])
    state = m.call([Int32, Int32, Int32, Int32, Int32, Int32, Int32, VarArgs(Int64)])
    assert(get_arg_gprs(state)[6:8] == ["arg06", "?"])
    assert(get_stack_objects(state) == ["varg00"])

    # a 2xlen argument with alignment less than 2xlen isn't passed in an
    # aligned register pair
    state = m.call([VarArgs(Int32, Struct(Ptr32, Int32))])
    assert(get_arg_gprs(state)[0:4] == ["varg00", "varg01[0:31]", "varg01[32:63]", "?"])

    # Floating point varargs are always passed according to the integer
    # calling convention
    state = m.call([Float, VarArgs(Double, Struct(Int32, Float))])
    assert(get_arg_gprs(state)[0:5] == ["varg00[0:31]", "varg00[32:63]",
        "varg01[0:31]", "varg01[32:63]", "?"])
    assert(get_arg_fprs(state)[0:2] == ["arg00", "?"])

    # Varargs should be promoted
    state = m.call([VarArgs(Float, Int8, UInt16)])
    assert([str(state.gprs[10]), str(state.gprs[11]), str(state.gprs[12])] ==
            ["FP32", "SInt32", "UInt32"])

def test_simple_usage():
    m = RVMachine(xlen=32, flen=64)
    state = m.call([
        Int32,
        Double,
        Struct(Int8, Array(Float, 1)),
        Struct(Array(Int8, 20)),
        Int64,
        Int64,
        Int64])
    assert(get_arg_gprs(state) == ["arg00", "arg02[0:7]", "&arg03", "arg04[0:31]", 
        "arg04[32:63]", "arg05[0:31]", "arg05[32:63]", "arg06[0:31]"])
    assert(get_arg_fprs(state)[0:3] == ["arg01", "arg02[32:63]", "?"])
    assert(len(state.stack) == 1)
    assert(state.typestr_or_name(state.stack[0]) == "arg06[32:63]")

def test_large_return():
    m = RVMachine(xlen=32)
    state = m.call([], Int128)
    assert(get_arg_gprs(state)[0:2] == ["&ret", "?"])
    state = m.call([], Int32)
    assert(get_arg_gprs(state)[0] == "?")
