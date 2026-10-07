from random import randint

from asciimatics.screen import Screen

# Write a line of code that imports the pyjokes package

# Write a line of code that imports the sleep function from the time module
# (which is already included in Python)

def demo(screen):
    while True:
        # Insert a line of code that randomly generates a joke using
        # pyjokes' get_joke() function
        
        # Change the below code to display the joke as the text that is
        # visualised
        screen.print_at("Hello world!",
                        randint(0, screen.width), randint(0, screen.height),
                        colour=randint(0, screen.colours - 1),
                        bg=randint(0, screen.colours - 1))
        ev = screen.get_key()
        if ev in (ord('Q'), ord('q')):
            return

        # Write a line of code that tells Python to go to sleep for 0.4
        # seconds, using the sleep() function of the time module


        screen.refresh()

Screen.wrapper(demo)