# Task 1

# Create a new empty list to store input numbers
list_of_numbers = []

# Loop 5 times, asking the user to input a number, casting it as integer, then appending to the list
for i in range(5):
  number_to_add = int(input("Please input a number: "))
  list_of_numbers.append(number_to_add)

# Use the sum() function to add up the list of numbers
sum_of_numbers = sum(list_of_numbers)

# Print the resultant sum
print (f"The total is {sum_of_numbers}")

# Task 2

# Create list of murder weapons
list_of_murder_weapons = ["shotgun", "handgun", "handgun", "metal urn", "iron gate", "pearl necklace", "driverless car", "bookend", "handgun", "knife"]

# Create a variable to store the number of guns
gun_count = 0

# For each weapon in the list of weapons, if the character sequence "gun" is in the name, add 1 to the gun count
for weapon in list_of_murder_weapons:
  if "gun" in weapon:
    gun_count += 1

# Print the total number of guns
print (f"There are {gun_count} guns in the list")

# Task 3

# Starting at the second element, up to the end of the list, print every second murder weapon
for i in range(1, len(list_of_murder_weapons), 2):
  print (list_of_murder_weapons[i])

# Task 4

# Ask the user to input their name
name = input("What is your name? ")

# Create a new string to store the backwards name
backwards_name = ""

# Starting at the last letter (index is length of string - 1), and going back up to
# and INCLUDING the last letter by one letter at a time, add the letter to the backwards name
for i in range(len(name)-1, -1, -1):
  backwards_name = backwards_name + name[i]

# Print out the backwards name
print (f"Your name spelled backwards is {backwards_name}")

