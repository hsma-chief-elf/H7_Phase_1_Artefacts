from cowpy import cow
import emoji

# Create a Ghostbusters cow
gb = cow.Ghostbusters()

# Milk the Ghostbusters cow to create a message, with an emoji
msg = gb.milk("Who you gonna call? :ghost:")

# Print the message with ghost emoji rendered
print (emoji.emojize(msg))

