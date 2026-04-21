def fibonacci():
    a = 0
    b = 1

    yield a
    yield b

    while True:
        fib_num = a + b
        yield fib_num

        a = b
        b = fib_num

my_fib_generator = fibonacci()

for x in range(100):
    print (next(my_fib_generator))

