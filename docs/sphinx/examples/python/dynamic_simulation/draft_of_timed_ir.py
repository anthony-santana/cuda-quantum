import cudaq


@cudaq.kernel
def kernel():
    qubit_0 = cudaq.qubit()
    qubit_1 = cudaq.qubit()

    h(qubit_0)
    h(qubit_1)

    x(qubit_0)
    x(qubit_1)

    x(qubit_0)
    h(qubit_1)


print(kernel)
"""

Note: This is pseudocode that I've spruced up for visualization.
      Much of this is "illegal" in MLIR-land but hopefully helps
      get the concepts across.


Original function:

```
  func.func @kernel() attributes {"cudaq-entrypoint"} {
    %0 = quake.alloca !quake.ref
    %1 = quake.alloca !quake.ref
    quake.h %0 : (!quake.ref) -> ()
    quake.h %1 : (!quake.ref) -> ()
    quake.x %0 : (!quake.ref) -> ()
    quake.x %1 : (!quake.ref) -> ()
    quake.x %0 : (!quake.ref) -> ()
    quake.h %1 : (!quake.ref) -> ()
    return
  }
```

New types:

  `quake.global_clock` : Operates in terms of unitless, discrete time steps.
                         Keeps track of the global scheduling for the "main"
                         kernel. 
                         At the hardware agnostic level, you could assume
                         all time step values are of the same physical time duration.
                         On hardware, however, the physical time duration of each `%global_time[step]`
                         may vary depending on which gates it is running.
                         I don't believe this should matter in our model, as we define
                         all time steps as being relative to one another.
  `quake.local_clock` or `quake.local_scope`: Blocks that contain quantum instructions which will
                                              run simultaneously.

                                              This could also just be an attribute on the existing
                                              op -- it maybe just help to visualize with "scopes"
                                              for now.

  `cudaq.block` or `cudaq.barrier` : A higher level type that allows a programmer to
                                     control which functions they'd like executed
                                     simultaneously.

Time scheduled Quake function:

```
  func.func @kernel() attributes {"cudaq-entrypoint"} {
    // Will initiate with the number of steps being the number
    // of unique sets of non-parallelizable instructions (pre-computed
    // in a previous step of the compiler.)
    // I'm not sure what computing this value may look like yet,
    // and would love to hear input...

    %global_time = quake.global_clock<NUMBER_OF_TIME_STEPS>

    %0 = quake.alloca !quake.ref
    %1 = quake.alloca !quake.ref
    
    // This local clock for the first layer of parallelizable operations
    // starts at the 0-th step of the `global_time`.

    quake.local_clock : %global_time[0] {
      quake.h %0 : (!quake.ref) -> ()
      quake.h %1 : (!quake.ref) -> ()
    }

    // This kicks off at the 1-st time step.

    quake.local_clock : %global_time[1] {
      quake.x %0 : (!quake.ref) -> ()
      quake.x %1 : (!quake.ref) -> ()
    }

    // Where it gets more interesting is when you have blocks
    // of operations that are parallelizable, but may be of
    // different duration on the hardware. In that case,
    // the total duration of the `local_clock` will be that of
    // the longest duration gate within it.
    // Before lowering to hardware, however, that shouldn't matter.
    // All that matters is that we can perform these two gates at the
    // same time.

    quake.local_clock : %global_time[2] {
      quake.x %0 : (!quake.ref) -> ()
      quake.h %1 : (!quake.ref) -> ()
    }

    // TODO:
    // 1. Think about two-qubit gates
    // 2. Think about more complex circuit and languge structures,
    //    such as while/for loops, and how those would play in here.

    return
  }
```

"""
