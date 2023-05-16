#!/usr/bin/env python

from kernel_tuner import tune_kernel
from kernel_tuner.util import (
    extract_directive_signature,
    extract_directive_code,
    extract_preprocessor,
    generate_directive_function,
    extract_initialization_code
)

with open("miniWeather_openacc.cpp") as file:
    code = file.read()

preprocessor = extract_preprocessor(code)
signatures = extract_directive_signature(code)
body = extract_directive_code(code)
init = extract_initialization_code(code)

for function in signatures.keys():
    kernel_string = generate_directive_function(
        preprocessor, signatures[function], body[function], init
    )

    tune_params = dict()
    tune_params["ngangs"] = [2**i for i in range(0, 13)]
    tune_params["vlength"] = [2**i for i in range(0, 11)]

    print(f"Tuning {function}")
    tune_kernel(
        function,
        kernel_string,
        0,
        [],
        tune_params,
        compiler_options=["-fast", "-acc=gpu", "-gpu=managed"],
        compiler="nvc++",
    )