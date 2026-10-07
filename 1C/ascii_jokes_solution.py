from random import randint

from asciimatics.screen import Screen

import pyjokes

from time import sleep

def demo(screen):
    while True:
        random_joke = pyjokes.get_joke()
        screen.print_at(random_joke,
                        randint(0, screen.width), randint(0, screen.height),
                        colour=randint(0, screen.colours - 1),
                        bg=randint(0, screen.colours - 1))
        ev = screen.get_key()
        if ev in (ord('Q'), ord('q')):
            return

        sleep(0.4)
        screen.refresh()

Screen.wrapper(demo)