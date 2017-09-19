#!/usr/bin/env python3
# Part of riscv-calling-conv-model 
# https://github.com/lowRISC/riscv-calling-conv-model
#
# See LICENSE file for copyright and license details

import copy, operator

# All alignments and sizes are currently specified in bits

def align_to(x, align):
    return x - (x % -align)

class Int(object):
    def __init__(self, size, signed=True):
        self.size = size
        self.alignment = size
        self.signed = signed
    def __repr__(self):
        return '{}Int{}'.format('S' if self.signed else 'U', self.size)

def UInt(size):
    return Int(size, signed=False)

def SInt(size):
    return Int(size, signed=True)

Int8, Int16, Int32, Int64, Int128 = [SInt(i) for i in [8,16,32,64,128]]
SInt8, SInt16, SInt32, SInt64, SInt128 = [Int8, Int16, Int32, Int64, Int128]
UInt8, UInt16, UInt32, UInt64, UInt128 = [UInt(i) for i in [8,16,32,64,128]]
Char = UInt8

class FP(object):
    def __init__(self, size):
        self.size = size
        self.alignment = size
    def __repr__(self):
        return 'FP{}'.format(self.size)

Float, Double, LongDouble = FP(32), FP(64), FP(128)
FP32, FP64, FP128 = Float, Double, LongDouble

class Ptr(object):
    def __init__(self, size):
        self.size = size
        self.alignment = size
    def __repr__(self):
        return 'Ptr{}'.format(self.size)

Ptr32, Ptr64 = Ptr(32), Ptr(64)

class Pad(object):
    def __init__(self, size):
        self.size = size
        self.alignment = 1
    def __repr__(self):
        return 'Pad{}'.format(self.size)

class Struct(object):
    # Add padding objects when necessary to ensure struct members have their 
    # desired alignment
    def add_padding(self):
        i = 0
        cur_offset = 0
        while i < len(self.members):
            wanted_align = self.members[i].alignment
            if (cur_offset % wanted_align) != 0:
                pad_size = -(cur_offset % -wanted_align)
                self.members.insert(i, Pad(pad_size))
                i += 1
                cur_offset += pad_size
            cur_offset += self.members[i].size
            i+= 1

    def __init__(self, *members):
        self.members = list(members)
        if len(members) == 0:
            self.alignment = 8
            self.size = 0
            return
        self.add_padding()
        self.alignment = max(m.alignment for m in members)
        self.size = sum(m.size for m in members)
        self.size = align_to(self.size, self.alignment)

    def flatten(self):
        children = []
        for ty in self.members:
            if hasattr(ty, 'flatten'):
                children += ty.flatten()
            else:
                children.append(ty)
        return children

    def __repr__(self):
        return 'Struct({}, s{}, a{})'.format(self.members, 
                self.size, self.alignment)

class Union(object):
    def __init__(self, *members):
        self.members = list(members)
        self.alignment = max(m.alignment for m in members)
        self.size = max(m.size for m in members)
        self.size = align_to(self.size, self.alignment)
    def __repr__(self):
        return 'Union({}, s{}, a{})'.format(self.members, 
                self.size, self.alignment)

class Array(object):
    def __init__(self, ty, num_elements):
        self.ty = ty
        self.num_elements = num_elements
        self.alignment = ty.alignment
        self.size = ty.size * num_elements

    def flatten(self):
        if hasattr(self.ty, 'flatten'):
            return self.ty.flatten() * self.num_elements
        else:
            return [self.ty] * self.num_elements

    def __repr__(self):
        return 'Array({}*{}, s{}, a{})'.format(self.ty, 
                self.num_elements, self.size, self.alignment)

class Slice(object):
    def __init__(self, child, low, high):
        self.child = child
        self.low = low
        self.high = high
        self.size = high - low + 1
        self.alignment = self.size
    def __repr__(self):
        return '{}[{}:{}]'.format(self.child, self.low, self.high)

class VarArgs(object):
    def __init__(self, *args):
        self.args = list(args)
    def __repr__(self):
        return 'VarArgs({})'.format(self.args)

class CCState(object):
    def __init__(self, xlen, flen, in_args, var_args_index, out_arg):
        self.xlen = xlen
        self.flen = flen
        self.gprs_left = 8
        self.gprs = [None] * 32
        self.fprs = None
        self.fprs_left = None
        if flen:
            self.fprs = [None] * 32
            self.fprs_left = 8
        self.stack = []
        self.type_name_mapping = {}
        self.in_args = in_args
        self.var_args_index = var_args_index
        self.out_arg = out_arg
        self.name_types(in_args, var_args_index, out_arg)

    def name_types(self, in_args, var_args_index, out_arg):
        i = 0
        arg_idx = 0
        varg_idx = 0
        for index, ty in enumerate(in_args):
            if index >= var_args_index:
                self.type_name_mapping[ty] = 'varg'+str(varg_idx).zfill(2)
                varg_idx += 1
            else:
                self.type_name_mapping[ty] = 'arg'+str(arg_idx).zfill(2)
                arg_idx += 1
        if out_arg:
            self.type_name_mapping[out_arg] = 'ret'

    def next_arg_gpr(self):
        return (8-self.gprs_left)+10

    def skip_gpr(self):
        if (self.gprs_left == 0):
            raise ValueError('all GPRs assigned')
        self.gprs_left -= 1

    def assign_to_gpr_or_stack(self, ty):
        if ty.size > self.xlen:
            raise ValueError('object is larger than xlen')
        if self.gprs_left >= 1:
            self.assign_to_gpr(ty)
        else:
            self.assign_to_stack(ty)

    def assign_to_gpr(self, ty):
        if ty.size > self.xlen:
            raise ValueError('object is larger than xlen')
        if self.gprs_left <= 0:
            raise ValueError('all argument registers already assigned')
        self.gprs[self.next_arg_gpr()] = ty
        self.gprs_left -= 1

    def next_arg_fpr(self):
        return (8-self.fprs_left)+10

    def assign_to_fpr(self, ty):
        if ty.size > self.flen:
            raise ValueError('object is larger than flen')
        if self.fprs_left <= 0:
            raise ValueError('all FP argument registers already assigned')
        self.fprs[self.next_arg_fpr()] = ty
        self.fprs_left -= 1

    def assign_to_stack(self, ty):
        if ty.size > 2*self.xlen:
            raise ValueError('objects larger than 2x xlen should be passed by reference')
        self.stack.append(ty)

    def pass_by_reference(self, ty):
        ptrty = Ptr(self.xlen)
        self.assign_to_gpr_or_stack(ptrty)
        if ty in self.type_name_mapping:
            self.type_name_mapping[ptrty] = '&'+self.type_name_mapping[ty]

    def typestr_or_name(self, ty):
        suffix = ''
        if ty == None:
            return '?'
        elif isinstance(ty, Slice):
            suffix = '[{}:{}]'.format(ty.low, ty.high)
            if ty.child in self.type_name_mapping:
                return self.type_name_mapping[ty.child]+suffix
            else:
                return repr(ty)
        return self.type_name_mapping.get(ty, repr(ty))

    def get_oldsp_rel_stack_locs(self):
        locs = []
        oldsp_off = 0
        for idx, ty in enumerate(self.stack):
            if idx == 0:
                locs.append(0)
                continue
            obj = self.stack[idx]
            prev_obj = self.stack[idx-1]
            oldsp_off += prev_obj.size
            oldsp_off = align_to(oldsp_off, self.xlen)
            oldsp_off = align_to(oldsp_off, obj.alignment)
            locs.append(oldsp_off//8)
        return locs

    def get_oldsp_rel_stack_loc(self, obj_idx):
        if obj_idx < 0 or obj_idx >= len(self.stack):
            raise ValueError("invalid stack object")
        obj = self.stack[obj_idx]
        sp_offset = 0
        for i in range(0, obj_idx):
            sp_offset = align_to(sp_offset, max(self.xlen, self.stack[i].alignment))
            sp_offset += self.stack[i].size
        sp_offset = align_to(sp_offset, max(self.xlen, obj.alignment))
        return sp_offset//8

    def __repr__(self):
        out = []
        if len(self.type_name_mapping) > 0:
            out.append('Args:')
            for item in sorted(self.type_name_mapping.items(), 
                    key=operator.itemgetter(1)):
                if item[1][0] == '&':
                    continue
                out.append('{}: {}'.format(item[1], item[0]))
            out.append('')
        out.append('GPRs:')
        for i in range(0, 8):
            out.append('GPR[a{}]: {}'.format(i, 
                self.typestr_or_name(self.gprs[i+10])))

        if self.flen:
            out.append('\nFPRs:')
            for i in range(0, 8):
                out.append('FPR[fa{}]: {}'.format(i,
                    self.typestr_or_name(self.fprs[i+10])))

        out.append('\nStack:')
        oldsp_offs = self.get_oldsp_rel_stack_locs()
        for idx, ty in enumerate(self.stack):
            out.append('{} (oldsp+{})'.format(self.typestr_or_name(ty),
                oldsp_offs[idx]))
        return '\n'.join(out)

class InvalidVarArgs(Exception):
    pass

class RVMachine(object):
    def __init__(self, xlen=64, flen=None):
        if xlen not in [32, 64, 128]:
            raise ValueError("unsupported XLEN")
        if flen and flen not in [32, 64, 128]:
            raise ValueError("unsupported FLEN")
        self.xlen = xlen
        self.flen = flen

    # Should be called after any expected VarArgs has been flattened
    def verify_arg_list(self, in_args, out_arg):
        # Ensure all argument/return type objects are unique
        if (len(in_args) != len(set(in_args))) or out_arg in in_args:
            raise ValueError("Unique type objects must be used")
        if isinstance(out_arg, VarArgs):
            raise InvalidVarArgs("Return type cannot be varargs")
        for arg in in_args:
            if (isinstance(arg, VarArgs)):
                raise InvalidVarArgs("VarArgs must be last element")

    def ret(self, ty):
        # Values are returned in the same way a named argument of the same
        # type would be passed. If it would be passed by reference, the
        # argument list is rewritten so the first argument is a pointer to
        # caller-allocated memory where the return value can be placed.
        in_args = []
        if ty:
            in_args = [ty]

        state = self.call(in_args)

        # Detect the case where the return value would be passed by reference
        if state.typestr_or_name(state.gprs[10]).startswith('&'):
            state.gprs[10] = None

        newty = next(iter(state.type_name_mapping))
        state.type_name_mapping[newty] = 'ret'
        return state


    def call(self, in_args, out_arg=None):
        # Remove the VarArgs wrapper type, but keep track of the arguments
        # specified to be vararg. var_args_index will point past the end of
        # in_args if there are no varargs.
        var_args_index = len(in_args)
        if len(in_args) >= 1 and isinstance(in_args[-1], VarArgs):
            var_args = in_args[-1].args
            in_args.pop()
            var_args_index = len(in_args)
            in_args.extend(var_args)

        # Ensure there's a unique object to represent every argument type
        in_args = [copy.copy(arg) for arg in in_args]
        out_arg = copy.copy(out_arg)

        self.verify_arg_list(in_args, out_arg)

        # Filter out empty structs
        in_args = [arg for arg in in_args if arg.size > 0]

        def isStruct(ty):
            return isinstance(ty, Struct)
        def isArray(ty):
            return isinstance(ty, Array)
        def isFP(ty):
            return isinstance(ty, FP)
        def isInt(ty):
            return isinstance(ty, Int)
        def isPad(ty):
            return isinstance(ty, Pad)

        xlen, flen = self.xlen, self.flen

        # Promote varargs
        for idx in range(var_args_index, len(in_args)):
            arg = in_args[idx]
            if isInt(arg) and arg.size < xlen:
                arg.size = xlen
                arg.alignment = xlen
            elif isFP(arg) and arg.size < xlen:
                arg.size = flen
                arg.alignment = flen

        state = CCState(xlen, flen, in_args, var_args_index, out_arg)

        # Error out if Arrays are being passed/returned directly. This isn't 
        # supported in C
        if isArray(out_arg) or any(isArray(ty) for ty in in_args):
            raise ValueError('Byval arrays not supported in C')

        # Catch the special case of returning a struct that can be returned 
        # according to the floating point calling convention
        if flen and isStruct(out_arg) and out_arg.size <= 2*flen:
            ty = Struct(*out_arg.flatten())
            mems = [mem for mem in ty.members if not isPad(mem)]
            if len(mems) == 2:
                ty1, ty2 = mems[0], mems[1]
                if ((isFP(ty1) and isFP(ty2) and
                    ty1.size <= flen and ty2.size <= flen) or
                   (isFP(ty1) and isInt(ty2) and
                    ty1.size <= flen and ty2.size <= xlen) or
                   (isInt(ty1) and isFP(ty2) and
                    ty1.size <= xlen and ty2.size <= flen)):
                    pass
                else:
                    state.pass_by_reference(out_arg)
        # If the return value won't be returned in registers, the address to 
        # store it to is passed as an implicit first parameter
        elif out_arg and out_arg.size > 2*xlen:
            state.pass_by_reference(out_arg)

        for index, ty in enumerate(in_args):
            is_var_arg = index >= var_args_index
            # Special-case rules introduced by the floating point calling 
            # convention
            if flen and not is_var_arg:
                # Flatten the struct if there is any chance it may be passed 
                # in fprs/gprs (i.e. it is possible it contains two floating 
                # point values, or one fp + one int)
                flat_ty = ty
                if isStruct(ty) and ty.size <= max(2*flen, 2*xlen):
                    flat_ty = Struct(*ty.flatten())
                    if len(flat_ty.members) == 1:
                        flat_ty = flat_ty[0]
                if isFP(flat_ty) and flat_ty.size <= flen and state.fprs_left >= 1:
                    state.assign_to_fpr(ty)
                    continue
                elif isStruct(flat_ty) and flat_ty.size <= 2*flen:
                    # Ignore any padding
                    mems = [mem for mem in flat_ty.members if not isPad(mem)]
                    if len(mems) == 2:
                        ty1, ty2 = mems[0], mems[1]
                        ty1_slice = Slice(ty, 0, ty1.size - 1)
                        ty2_off = max(ty1.size, ty2.alignment)
                        ty2_slice = Slice(ty, ty2_off,
                                          ty2_off + ty2.size - 1)
                        if (isFP(ty1) and isFP(ty2)
                            and ty1.size <= flen and ty2.size <= flen
                            and state.fprs_left >= 2):
                           state.assign_to_fpr(ty1_slice)
                           state.assign_to_fpr(ty2_slice)
                           continue
                        elif (isFP(ty1) and isInt(ty2) and
                              ty1.size <= flen and ty2.size <= xlen and
                              state.fprs_left >= 1 and state.gprs_left >= 1):
                            state.assign_to_fpr(ty1_slice)
                            state.assign_to_gpr(ty2_slice)
                            continue
                        elif (isInt(ty1) and isFP(ty2) and
                              ty1.size <= xlen and ty2.size <= flen and
                              state.gprs_left >=1 and state.fprs_left >=1):
                            state.assign_to_gpr(ty1_slice)
                            state.assign_to_fpr(ty2_slice)
                            continue

            # If we got to here, the standard integer calling convention 
            # applies
            if ty.size <= xlen:
                state.assign_to_gpr_or_stack(ty)
            elif ty.size <= 2*xlen:
                # 2xlen-aligned varargs must be passed in an aligned register
                # pair
                if (is_var_arg and ty.alignment == 2*xlen
                    and state.gprs_left % 2 == 1):
                    state.skip_gpr()
                if state.gprs_left > 0:
                    state.assign_to_gpr_or_stack(Slice(ty, 0, xlen-1))
                    state.assign_to_gpr_or_stack(Slice(ty, xlen, 2*xlen - 1))
                else:
                    state.assign_to_stack(ty)
            else:
                state.pass_by_reference(ty)
        return state

if __name__ == '__main__':
    print("""
Usage example:
$ python3
Python 3.6.2 (default, Jul 20 2017, 03:52:27)
[GCC 7.1.1 20170630] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from rvcc import *
>>> m = RVMachine(xlen=32, flen=64)
>>> m.call([
... Int32,
... Double,
... Struct(Int8, Array(Float, 1)),
... Struct(Array(Int8, 20)),
... Int64,
... Int64,
... Int64])
Args:
arg00: SInt32
arg01: FP64
arg02: Struct([SInt8, Pad24, Array(FP32*1, s32, a32)], s64, a32)
arg03: Struct([Array(SInt8*20, s160, a8)], s160, a8)
arg04: SInt64
arg05: SInt64
arg06: SInt64

GPRs:
GPR[a0]: arg00
GPR[a1]: arg02[0:7]
GPR[a2]: &arg03
GPR[a3]: arg04[0:31]
GPR[a4]: arg04[32:63]
GPR[a5]: arg05[0:31]
GPR[a6]: arg05[32:63]
GPR[a7]: arg06[0:31]

FPRs:
FPR[fa0]: arg01
FPR[fa1]: arg02[32:63]
FPR[fa2]: ?
FPR[fa3]: ?
FPR[fa4]: ?
FPR[fa5]: ?
FPR[fa6]: ?
FPR[fa7]: ?

Stack:
arg06[32:63] (oldsp+0)
""")
