# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Huawei KMC library."""

import ctypes
import os
import random
from ctypes.util import find_library
import logging
import platform

__all__ = ["init", "encrypt", "decrypt", "check_and_update_mk", "update_root_key", "hmac", "hmac_verify", "finalize"]

_kmc_dll: ctypes.CDLL = None
_libc_dll: ctypes.CDLL = None
ADVANCE_DAY = 3

def hmac(domain_id: int, plain_text: str) -> str:
    """Encode HMAC code."""
    p_char = ctypes.c_char_p()
    hmac_len = ctypes.c_int(0)
    c_plain_text = ctypes.create_string_buffer(plain_text.encode())
    _kmc_dll.KeHmacByDomain.restype = ctypes.c_int
    _kmc_dll.KeHmacByDomain.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int)]
    ret = _kmc_dll.KeHmacByDomain(
        domain_id, c_plain_text, len(plain_text), ctypes.byref(p_char), ctypes.pointer(hmac_len))
    if ret != 0:
        logging.error(f"failed to call KeHmacByDomain, code={ret}")
    value = p_char.value.decode()
    ret = _libc_dll.free(p_char)
    if ret != 0:
        logging.error(f"failed to free resource, code={ret}")
    return value


def hmac_verify(domain_id: int, plain_text: str, hmac_text: str) -> bool:
    """Verify HMAC code."""
    c_plain_text = ctypes.create_string_buffer(plain_text.encode())
    c_hmac_text = ctypes.create_string_buffer(hmac_text.encode())
    _kmc_dll.KeHmacVerifyByDomain.restype = ctypes.c_int
    _kmc_dll.KeHmacVerifyByDomain.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    ret = _kmc_dll.KeHmacVerifyByDomain(domain_id, c_plain_text, len(plain_text), c_hmac_text, len(c_hmac_text))
    return ret


def encrypt(domain_id: int, plain_text: str) -> str:
    """Encrypt."""
    p_char = ctypes.c_char_p()
    cipher_len = ctypes.c_int(0)
    c_plain_text = ctypes.create_string_buffer(plain_text.encode())

    _kmc_dll.KeEncryptByDomain.restype = ctypes.c_int
    _kmc_dll.KeEncryptByDomain.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
                                           ctypes.POINTER(ctypes.c_int)]
    ret = _kmc_dll.KeEncryptByDomain(domain_id, c_plain_text, len(plain_text), ctypes.byref(p_char),
                                     ctypes.pointer(cipher_len))
    if ret != 0:
        logging.error("KeEncryptByDomain failed.")
        return ""
    value = p_char.value.decode()
    ret = _libc_dll.free(p_char)
    if ret != 0:
        logging.error("free memory error. ret=%d" % ret)
    return value


def _decrypt(domain_id: int, cipher_text: str):
    """Decrypt."""
    p_char = ctypes.c_char_p()
    plain_len = ctypes.c_int(0)
    c_cipher_text = ctypes.create_string_buffer(cipher_text.encode())
    _kmc_dll.KeDecryptByDomain.restype = ctypes.c_int
    _kmc_dll.KeDecryptByDomain.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
                                           ctypes.POINTER(ctypes.c_int)]
    ret = _kmc_dll.KeDecryptByDomain(domain_id, c_cipher_text, len(cipher_text), ctypes.byref(p_char),
                                     ctypes.pointer(plain_len))
    if ret != 0:
        logging.error("KeDecryptByDomain failed.")
        return ""
    value = p_char.value.decode()
    ret = _libc_dll.free(p_char)
    if ret != 0:
        logging.error("free memory error. ret=%d" % ret)
    return value


def check_and_update_mk(domain_id: int, advance_day: int) -> bool:
    """Check and update mk."""
    try:
        _kmc_dll.KeRefreshMkMask()
    except Exception as err:
        logging.error('refresh_task failed, catch error: %s', err)
    
    ret = _kmc_dll.KeCheckAndUpdateMk(domain_id, advance_day)
    if ret != 0:
        logging.error(f"failed to call KeCheckAndUpdateMk, code={ret}")
        return False
    return True


def update_root_key() -> bool:
    """Update root key."""
    ret = _kmc_dll.KeUpdateRootKey()
    if ret != 0:
        logging.error(f"failed to call KeUpdateRootKey, code={ret}")
        return False
    return True


def finalize() -> None:
    """Finalize."""
    _kmc_dll.KeFinalize.restype = ctypes.c_int
    _kmc_dll.KeFinalize.argtypes = []
    _kmc_dll.KeFinalize()


def _get_lib_path():
    pkg_path = os.path.dirname(__file__)
    if platform.processor() == "x86_64":
        return os.path.join(pkg_path, "x86_64/libkmcext.so")
    else:
        return os.path.join(pkg_path, "aarch64/libkmcext.so")


def _load_dll(kmc_dll_path: str) -> None:
    global _kmc_dll
    if _kmc_dll:
        return
    global _libc_dll
    if _libc_dll:
        return
    _libc_dll = ctypes.CDLL(find_library("c"))
    _kmc_dll = ctypes.CDLL(kmc_dll_path)


@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
def _logger(level: ctypes.c_int, msg: ctypes.c_char_p):
    logging.info("level:%d, msg:%s" % (level, str(msg)))


def _init_log():
    _kmc_dll.KeSetLoggerCallback.restype = None
    _kmc_dll.KeSetLoggerCallback.argtypes = [ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)]
    _kmc_dll.KeSetLoggerCallback(_logger)
    _kmc_dll.KeSetLoggerLevel.restype = None
    _kmc_dll.KeSetLoggerLevel.argtypes = [ctypes.c_int]
    _kmc_dll.KeSetLoggerLevel(2)  # DISABLE(0),ERROR(1),WARN(2),INFO(3),DEBUG(4),TRACE(5)


class KMCConfig(ctypes.Structure):
    _fields_ = [
        ("primaryKeyStoreFile", ctypes.c_char * 4096),
        ("standbyKeyStoreFile", ctypes.c_char * 4096),
        ("domainCount", ctypes.c_int),
        ("role", ctypes.c_int),
        ("procLockPerm", ctypes.c_int),
        ("sdpAlgId", ctypes.c_int),
        ("hmacAlgId", ctypes.c_int),
        ("semKey", ctypes.c_int)
    ]


def _init_kmc_config(primary_key_store_file, standby_key_store_file, alg_id, domain_count):
    config = KMCConfig()
    config.primaryKeyStoreFile = primary_key_store_file.encode()
    config.standbyKeyStoreFile = standby_key_store_file.encode()
    config.domainCount = domain_count
    config.role = 1  # Agent 0; Master 1
    config.procLockPerm = 0o0600
    config.sdpAlgId = alg_id
    config.hmacAlgId = 2052  # HMAC_SHA256 2052; HMAC_SHA384 2053 HMAC_SHA512 2054
    DEFAULT_SEM_KEY = 0x20160000
    MIN_HEX_SEM_KEY = 0x1111
    MAX_HEX_SEM_KEY = 0x9999
    config.semKey = DEFAULT_SEM_KEY + \
        random.randint(MIN_HEX_SEM_KEY, MAX_HEX_SEM_KEY)
    _kmc_dll.KeInitialize.restype = ctypes.c_int
    _kmc_dll.KeInitialize.argtypes = [ctypes.POINTER(KMCConfig)]
    return _kmc_dll.KeInitialize(ctypes.byref(config))


def init(primary_key_store_file: str, standby_key_store_file: str, alg_id: int, domain_count=3) -> bool:
    """Initialize."""
    if alg_id not in [8, 9]:  # AES128_GCM, AES256_GCM
        logging.error(f"alg (id={alg_id}) is not legal")
        return False
    _load_dll(_get_lib_path())
    _init_log()
    ret = _init_kmc_config(primary_key_store_file, standby_key_store_file, alg_id, domain_count)
    if ret != 0:
        logging.error(f"failed to call KeInitialized, code={ret}")
        return False
    domain_id = 0
    try:
        _kmc_dll.KeActiveNewKey(domain_id)
    except Exception:
        logging.error("failed to call KeActiveNewKey.")

    check_and_update_mk(domain_id, ADVANCE_DAY)
    return True


def decrypt(cert_pem_file, secret_key_file, key_mm, key_component_1, key_component_2):
    """Decrypt the passwd."""
    sdp_alg_id = 9
    # Make sure ssl certificate file exist
    ca_file_list = (cert_pem_file, secret_key_file)
    for file in ca_file_list:
        if file and os.path.exists(file):
            continue
        else:
            logging.error("SSL Certificate files does not exist! Please check config.yaml and cert file.")
            raise FileNotFoundError

    primary_keyStoreFile = key_component_1
    standby_keyStoreFile = key_component_2
    ret = init(primary_keyStoreFile, standby_keyStoreFile, sdp_alg_id)
    if ret is False:
        logging.error("kmc init error.")
        raise Exception('ERROR: kmc init failed!')
    domain_id = 0
    decrypt_mm = _decrypt(domain_id, key_mm)
    if decrypt_mm == "":
        logging.error("kmc init error.")
        raise Exception('ERROR: kmc init failed!')
    finalize()
    return decrypt_mm
