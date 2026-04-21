
# We define a generator function in the same way as a conventional function
def keep_count_generator():
    # We'll set count to 0
    count = 0

    # Keep doing this indefinitely
    while True:
        # Add 1 to the count
        count += 1

        # The 'yield' statement identifies this as a generator function.
        # Yield is like return, but it tells the generator function to freeze
        # in place and remember where it was ready for the next time it's
        # called
        yield count

# We run a generator function a little differently than a conventional
# function.  Here, we create an 'instance' of the generator function, and then
# we can work with that instance.  This also means we can have multiple
# instances of the same generator function, all in different "states"
my_generator = keep_count_generator()

# Let's print the output of the generator function 5 times.  The 'next'
# statement is used to move to the next element of the iterator.  Here, that
# means the generator unfreezes and carries on until it hits another yield
# statement.
# I've not put these print statements in a for loop to make it clear what's
# happening.
print (next(my_generator))
print (next(my_generator))
print (next(my_generator))
print (next(my_generator))
print (next(my_generator))

