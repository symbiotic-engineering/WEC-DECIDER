# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 03:52:43 2023

@author: rgm222
"""

import autograd.numpy as np
import xarray as xr
import time
import functools
import dask

ds = xr.tutorial.open_dataset('rasm')
b = ds.isel(y=0)

class HashWrapper:
    def __init__(self, x) -> None:
        self.value = x 
        with dask.config.set({"tokenize.ensure-deterministic":True}):
            self.h = dask.base.tokenize(x)
    def __hash__(self) -> int:
        return hash(self.h)
    def __eq__(self, __value: object) -> bool:
        return __value.h == self.h

def hashable_cache(function):
    @functools.cache
    def cached_wrapper(*args, **kwargs):
        arg_values = [a.value for a in args]
        kwargs_values = {
            k: v.value for k,v in kwargs.items()
        }
        return function(*arg_values, **kwargs_values)
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        shell_args = [HashWrapper(a) for a in args]
        shell_kwargs = {
            k: HashWrapper(v) for k,v in kwargs.items()
        }
        return cached_wrapper(*shell_args, **shell_kwargs)
    
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear
    
    return wrapper

@hashable_cache
def myfunc(x):
    time.sleep(2)
    return x

@hashable_cache
def myfunc2(x,y):
    time.sleep(2)
    return x+y

t1 = time.time()

class TestClass:
    def __init__(self,x) -> None:
        self.x = xs
        
# myobj = TestClass(4)
# with dask.config.set({"tokenize.ensure-deterministic":True}):
#     dask.base.tokenize(myobj)


# y = myfunc(b)
# print(y)

# y2 = myfunc(b)
# print(y2)

b = np.array([1,2,3])

y = myfunc2(b,b)
y2 = myfunc2(b,b)

# modify foo's data
#b['Tair']= b['Tair'] * 1.1
b[0] = 2

# y3 = myfunc(b)
# print(y3)
y3 = myfunc2(b,b)

t2 = time.time()

print('Elapsed: ',t2-t1)

