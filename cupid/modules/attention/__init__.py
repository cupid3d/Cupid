from typing import *

BACKEND = 'flash_attn'
BACKEND_WITH_ATTN_BIAS = 'sdpa'
DEBUG = False

def __from_env():
    import os
    
    global BACKEND
    global BACKEND_WITH_ATTN_BIAS
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_backend_with_bias = os.environ.get('ATTN_BACKEND_WITH_ATTN_BIAS')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    if env_attn_backend_with_bias is not None and env_attn_backend_with_bias in ['sdpa', 'naive']:
        BACKEND_WITH_ATTN_BIAS = env_attn_backend_with_bias
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    print(f"[ATTENTION] Using backend: {BACKEND}, backend with attn bias: {BACKEND_WITH_ATTN_BIAS}")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn', 'sdpa', 'naive']):
    global BACKEND
    BACKEND = backend

def set_backend_with_attn_bias(backend: Literal['sdpa', 'naive']):
    global BACKEND_WITH_ATTN_BIAS
    BACKEND_WITH_ATTN_BIAS = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


from .full_attn import *
from .modules import *
