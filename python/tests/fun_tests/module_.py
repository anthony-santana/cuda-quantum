import inspect
import types 


class ModuleRunner:

    import numpy as np
    
    def __init__(self, function, *args, **kwargs):
        self.function = function 
        self.input_args = args 
        self.input_kwargs = kwargs 

        # Define the necessary imports to run the function.
        self.function_source = """""" # """import numpy as np\n"""
        # Get the function source code as a string.
        inspect_source = inspect.getsource(function)
        # Strip out the decorator and insert the rest of the function source.
        self.function_source += inspect_source.split("@module_.ModuleRunner\n")[1]

    def __call__(self, *args, **kwargs):
        """Call `self.function`."""
        print(self.function_source)
        print("\nExecuting the function!\n")
        # exec(self.function_source)

        # Seems like this method still requires the source
        # file to have certain imports since we're returning
        # the actual function from the file here.
        def executable(function_string):
            globals = dict()
            locals = dict()
            exec(function_string, globals, locals)
            # print(dir(globals), "\n\n")
            # print(list(globals.values()), "\n\n")

            # def new_new_foo(*args, **kwargs): pass
            # new_new_foo.__code__ = list(locals.values())[0].__code__
            # return new_new_foo
        
            # print(list(locals.values()))
            return list(locals.values())[0]
        
        new_foo = executable(self.function_source)
        print(new_foo)
        print(new_foo(5,1.))


        # # Compile into a code object.
        # code_object = compile(self.function_source, '<string>', 'exec')
        # print(type(code_object))
        # print(dir(code_object), "\n\n")
        # # print(code_object.co_consts)
        # executable = types.FunctionType(code_object, globals())
        # executable()



def test_test():
  source = """import math\nprint(math.pi)\n"""
  exec(source, globals(), locals())
  print(globals())


# test_test()