from typing import Optional, Union

def process_value(a: Optional[Union[int, float]] = 2) -> float:
    if a is None:
        print("Value is None, defaulting to 0.")
        a = 0
    result = a * 2.5
    print(f"Processed value: {result}")
    return result

# Examples of calling the function:

# Using the default value
process_value()

# Passing an integer
process_value(4)

# Passing a float
process_value(3.5)

# Passing None
process_value(None)
