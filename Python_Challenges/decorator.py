@my_decorator
def say_hello():
  print("Hello")

# short cut 
def say_hello():
  print("Hello!")
say_hello = my_decorator(say_hello)



