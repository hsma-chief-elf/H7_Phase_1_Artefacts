# Task 1

# Set answer variable to 0
answer = 0

# While answer is less than 1000, keep asking the user what number they're thinking of
while answer < 1000:
  answer = float(input("What number are you thinking of now? "))

# This code will only run once the user inputs a number of 1000 or more
print ("That's a big number!")

# Task 2

# Set countdown timer to 5
countdown_timer = 5

# While timer is above 0, print the current value and decrement by 1
while countdown_timer > 0:
  print (countdown_timer)
  countdown_timer -= 1

# Print Thunderbirds are go!
print ("Thunderbirds are go!")

# Task 3

# Create new empty list of tv shows
list_of_best_tv_shows = []

# Keep asking the user for a great TV show and adding them to the list until they either
# enter Murder She Wrote or they've given 5 answers
while "murder she wrote" not in list_of_best_tv_shows and len(list_of_best_tv_shows) < 5:
  tv_show = input("Name a great TV show: ")
  list_of_best_tv_shows.append(tv_show.lower())

# Print the list
print (list_of_best_tv_shows)

# Task 4

# Do this infinitely
while True:
  # Print welcome message
  print ("Let's start tracking your training hours")

  # Set total variable value to 0
  total = 0

  # While the total is less than 100, keep asking the user how many hours and add this to the total
  while total < 100:
    hours = int(input("How many hours training have you done this week? "))
    total += hours

  # Once 100 hours has been met (or exceeded) print message confirming the course is complete
  print ("Course complete!")

  # Ask the user if they want to track another course
  move_on = input("Track another one (y/n)? ")

  # If the user chooses no, then break out of the outer while loop
  if move_on.lower() == "n":
    break

