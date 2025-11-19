@my_decorator
def say_hello():
  print("Hello")

# short cut 
def say_hello():
  print("Hello!")
say_hello = my_decorator(say_hello)

# Decorators with arguments

@app.route("/")
@app.route("/hello/<name>")
@app.errorhandler(404)

# Each of them registers a function with Flask


