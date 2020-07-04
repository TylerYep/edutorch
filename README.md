# edutorch
Rewritten PyTorch framework designed to help you learn AI/ML



# Dynamic Shape-Checking / Type-Checking (Typorch)
I'm interested in shape-checking for tensors. It doesnt need to be static type checking,
having fancy asserts would be good enough.

> (B, C, H, W)

> (B, 1, H, W)

> (B, H, W)

If you run a program with the shape-checker, it automatically inserts assert statements that the left side of the variable assignment must have that shape. Letters are tracked throughout (e.g. a new letter introduces a new variable), and a number asserts that that dimension must match exactly.

Tuple shapes maybe, to distinguish a shape comment from a regular comment.

Using this mode, the code is compiled uniquely and increases runtime.

Once you are confident with your shapes, you can simply run your program normally.

# Goals
1. Readability. Everything should make it immediately obvious how the layer or mmodel works on its own.
2.


No autograd - if you want a simple autograd implementation, check out Karpathy's micrograd repo.

TODO:

- Convert to dataclasses?