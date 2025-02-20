import re



def change_interval (x, inmin, inmax, outmin, outmax):
    # making sure x is in the interval
    x = max(inmin, min(inmax, x))
    # normalizing x between 0 and 1
    x = (x - inmin) / (inmax - inmin)
    # denormalizing between outmin and outmax
    return x * (outmax - outmin) + outmin

def change_float_to_int_interval (x, inmin, inmax, outdiscmin, outdiscmax):
    x = change_interval(x, inmin, inmax, 0, 1)
    return round(x * (outdiscmax - outdiscmin) + outdiscmin)

def prefix_to_infix(expr, instructions , symbols, arity):
    expr = expr.strip().rstrip(';')  # Remove trailing semicolon if present
    
    # Base case: If it's just a variable or number, return it as-is
    if expr.isalnum():
        return expr

    # Match function-like expressions: func(arg1, arg2)
    match = re.match(r'(\w+)\((.*)\)', expr)
    if not match:
        return expr  # Return as-is if it's not a function call

    operator = match.group(1)  # Function name (e.g., sum, mult, sin)
    inner_expr = match.group(2)  # Arguments inside parentheses

    # Split arguments while handling nested parentheses
    args = []
    balance = 0
    start = 0

    for i, char in enumerate(inner_expr):
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        elif char == ',' and balance == 0:
            args.append(inner_expr[start:i].strip())
            start = i + 1

    args.append(inner_expr[start:].strip())  # Add last argument

    # Recursively process arguments
    args = [prefix_to_infix(arg, instructions, symbols, arity) for arg in args]


    
    # Convert based on known operators
    if operator in instructions and len(args) == 2:
        # print(instructions.index(operator))
        return f"({args[0]} {symbols[instructions.index(operator)]} {args[1]})"
    elif len(args) == 1:
        return f"{operator}({args[0]})"
    
    # If function is unknown, return as-is
    return f"{operator}({', '.join(args)})"