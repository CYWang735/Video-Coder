"""Provide os.memfd_create on Python builds that lack it.

conda-forge's CPython is compiled against an old glibc sysroot, so os.memfd_create and
os.MFD_* are missing even though the node's glibc (2.34) has memfd_create. SGLang's MiniMax-H3
reference-video decoder calls os.memfd_create(name, flags=os.MFD_CLOEXEC) unconditionally.
setup_tmp_env.sh installs this module plus a .pth file so it runs in every interpreter of the env.
"""
import os

if not hasattr(os, "memfd_create") and os.uname().sysname == "Linux":
    import ctypes

    _libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(_libc, "memfd_create"):
        _libc.memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
        _libc.memfd_create.restype = ctypes.c_int

        def memfd_create(name, flags=1):
            fd = _libc.memfd_create(os.fsencode(name), flags)
            if fd < 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err))
            return fd

        os.memfd_create = memfd_create
        for _name, _value in {"MFD_CLOEXEC": 0x1, "MFD_ALLOW_SEALING": 0x2, "MFD_HUGETLB": 0x4}.items():
            if not hasattr(os, _name):
                setattr(os, _name, _value)
