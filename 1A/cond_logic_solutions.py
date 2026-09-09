# Task 1

# Ask the user to input their name
name = (input("What is your name? "))

# If their name is Dan or Sammi, print the appropriate message
# Otherwise print the message that you don't recognise them
# Convert name to lowercase for check
if name.lower() == "dan" or name.lower() == "sammi":
  print (f"Hi {name}! You're a HSMA trainer")
else:
  print ("I don't recongise you")

# Task 2

# Ask the user to input their age
age = int(input("How old are you? "))

# If they're at least 18, ask what course they're doing
if age >= 18:
  course_name = input("What is the name of your course? ")

  # If the course is HSMA, welcome them to the programme
  if course_name.lower() == "hsma":
    print ("Welcome to the HSMA programme!")

# Task 3

# Ask the user to input two numbers, and store them as floats
num_1 = float(input("Please enter the first number: "))
num_2 = float(input("Please enter the second number :"))

# Add the numbers together and store them in total.  Then print
# the answer
total = num_1 + num_2
print (f"The answer is {total}")

# If the total is less than 10, tell them to think bigger.  If it's
# 1000 or more, tell them to think smaller.  Otherwise, tell them
# their thinking is just about right.
if total < 10:
  print ("You need to think bigger!")
elif total >= 1000:
  print ("You need to think smaller!")
else:
  print ("Your thinking is just about right")

# Task 4

# Get inputs from user, cast as integers and store in variables
monthly_take_home = int(input("What's your monthly take home income? : "))
housing_costs = int(input("What's your monthly housing cost (rent/mortgage)? "))
food_costs = int(input("How much do you spend on food per month? "))
utility_costs = int(input("How much do you spend on utilities per month? "))

# Calculate remaining money after housing, food and utility costs
remaining = monthly_take_home - housing_costs - food_costs - utility_costs

# Calculate the percentage of take home that housing costs represent
housing_perc = housing_costs / monthly_take_home

# Set a boolean depending on whether the housing costs make up more than 50% of take home
if housing_perc > 0.5:
  housing_perc_over_50 = True
else:
  housing_perc_over_50 = False

# Print the message to the user
print (f"Your monthly amount after housing, food and utility costs is £{remaining}.  Your housing costs represent {housing_perc*100:.2f}% of your monthly take home.")

if housing_perc_over_50 == True:
  print ("You are severely cost burdened by your housing costs")

# Task 5

# Ask the user to input a day number.
input_day = int(input("Please input a day from 1 to 365 : "))

# Use modulus to check the remainder when dividing the day number by 7. This
# will give a remainder of 1 for the first day, 2 for the second day etc, and
# will loop back around to give a remainder of 1 for the 8th day, 9 for the 9th
# day etc.
if input_day % 7 == 1:
    print ("Monday")
elif input_day % 7 == 2:
    print ("Tuesday")
elif input_day % 7 == 3:
    print ("Wednesday")
elif input_day % 7 == 4:
    print ("Thursday")
elif input_day % 7 == 5:
    print ("Friday")
elif input_day % 7 == 6:
    print ("Saturday")
elif input_day % 7 == 0:
    print ("Sunday")
else:
    print ("Erm... something's gone wrong!")