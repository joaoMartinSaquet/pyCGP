# import re

# # Mapping of function names to symbols
# instruction_map = {
#     'sum': '+',
#     'mul': '*',
#     'sin': 'sin',  # Keeping sin as sin, or you can extend this
#     'cos': 'cos',  # Same for other functions if needed
#     # Add more function mappings as needed
# }

# # Function to transform the expression
# def transform_expression(expr, instruction, sympy_symbols, arity):
#     if arity == 2:
#         while instruction in expr:
#             len_inst = len(instruction)
#             ind_inst = expr.find(instruction) + len_inst

            
#             pattern = r'sum\((.*?),\s*(.*?)\)'

#             ind_g1 = 0
#             ind_g2 = 0
#             gb = expr[0:expr.find(instruction)]
#             g1 = ''
#             g2 = ''
            
#             for i in range(ind_inst,len(expr)):
                    

#                 if expr[i] == ',':

#                     sexpr = expr[i].split(',')
#                     bexpr = sexpr[0]
#                     pexpr = sexpr[1]

#                     opening = expr[ind_inst:i].count('(')
#                     closing = expr[i+1:].count(')') 
#                     if closing == opening:
#                         g1 = expr[ind_inst:i]
#                         g2 = expr[i+1:]
#                     break

#             match = re.search(pattern, expr)
#             expr = gb + g1 + sympy_symbols + g2
#     else :
#         expr.replace(instruction, sympy_symbols)
#     return expr.replace(' ', '')


# def prefix_to_infix(expression):
#     stack = []
#     operators = set(['+', '-', '*', '/'])

#     # Split the expression into tokens
#     tokens = expression.split()[::-1]  # Reverse for easier processing

#     for token in tokens:
#         if token in operators:
#             # Pop two operands
#             operand1 = stack.pop()
#             operand2 = stack.pop()
#             # Form an infix expression
#             new_expr = f"({operand1} {token} {operand2})"
#             # Push the new expression back onto the stack
#             stack.append(new_expr)
#         else:
#             # Push operands directly to stack
#             stack.append(token)

#     return stack[0]  # Final result


# # Example input
# expression = 'y=sum(mul(x,x),x)'
# # expression = 'sum(x, sin(mult(x, mult(x, x))));)'

import re

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
        print(instructions.index(operator))
        return f"({args[0]} {symbols[instructions.index(operator)]} {args[1]})"
    elif len(args) == 1:
        return f"{operator}({args[0]})"
    
    # If function is unknown, return as-is
    return f"{operator}({', '.join(args)})"

instr = ['sum', 'mult', 'sin', 'cos']
symbols = ['+', '*', 'sin', 'cos']
arity = [2, 2, 1, 1]

# Example Usage
prefix_expr = "sum(sin(mult(mult(x, x), x)), x);"
prefix_expr = "y = sum(sin(mult(x, mult(x, x))), x)"
infix_expr = prefix_to_infix(prefix_expr, instr, symbols, arity)
print(infix_expr)  # Output: x + sin(x * (x * x))
