def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        result = func(*args, **kwargs)
        print("Something is happening after the function is called.")
        return result

    return wrapper


class MyClass:
    def my_method(self):
        print("The original method is called.")


# Create an instance of the class
obj = MyClass()

# Access the method
original_method = obj.my_method

# Apply the decorator
decorated_method = my_decorator(original_method)

# Reassign the decorated method back to the object
obj.my_method = decorated_method

# Call the decorated method
obj.my_method()
