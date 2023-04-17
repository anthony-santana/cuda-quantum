# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from _pycudaq import *
from typing import List
import sys, subprocess, os
import ast, inspect

class MidCircuitMeasurementAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.measureResultsVars = []
        self.hasMidCircuitMeasures = False

    def visit_Assign(self, node):
        target = node.targets[0]
        creatorFunc = node.value.func
        if 'id' in creatorFunc.__dict__ and creatorFunc.id == 'mz':
            self.measureResultsVars.append(target.id)
    
    def visit_If(self, node):
        condition = node.test
        if 'id' in condition.__dict__ and condition.id in self.measureResultsVars: 
            self.hasMidCircuitMeasures = True

initKwargs = {'qpu': 'qpp', 'platform':'default'}

if '-qpu' in sys.argv:
    initKwargs['qpu'] = sys.argv[sys.argv.index('-qpu')+1]

if '--qpu' in sys.argv:
    initKwargs['qpu'] = sys.argv[sys.argv.index('--qpu')+1]

if '-platform' in sys.argv:
    initKwargs['platform'] = sys.argv[sys.argv.index('-platform')+1]

if '--platform' in sys.argv:
    initKwargs['platform'] = sys.argv[sys.argv.index('--platform')+1]

initialize_cudaq(**initKwargs)

# We will have to list out all of our
# instructions like this.
h = h()
x = x()
y = y()
z = z()
s = s()
t = t()


class kernel(object):
    def __init__(self, function, *args, **kwargs):
        self.kernelFunction = function
        self.inputArgs = args 
        self.inputKwargs = kwargs 
        self.funcSrc = inspect.getsource(function)
        self.module = ast.parse(self.funcSrc)
        analyzer = MidCircuitMeasurementAnalyzer()
        analyzer.visit(self.module)
        self.metadata = {'conditionalOnMeasure': analyzer.hasMidCircuitMeasures}
    
    def __call__(self, *args):
        self.kernelFunction(*args)