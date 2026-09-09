# PART 1

# Ask the user's name and store in a variable
name = input("What's your name? ")

# Ask the user's age and store in a variable
age = input("What's your age? ")

# Cast the input age as an integer
age_as_int = int(age)

# Print a message greeting the user with their name and telling them
# how old they'll be in 10 years.
print (f"Hello {name}.  You'll be {age_as_int + 10} in 10 years time")

# PART 2

# Ask user to input diameter of pizza
pizza_diameter = int(input("What is the size of your pizza in inches? "))

# Ask user how many people there are
number_of_people = int(input("How many people are sharing this pizza? "))

# Calculate the radius of the pizza
pizza_radius = pizza_diameter / 2

# Calculate the area of the pizza
pizza_area = 3.14 * (pizza_radius**2)

# Calculate the amount of pizza each person gets
pizza_area_per_person = pizza_area / number_of_people

# Print the amount of pizza per person
print (f"Each person should get {pizza_area_per_person:.1f} square inches of pizza")

# Calculate the amount that would be left spare if the pizza was cut into slices of 10 square inches
pizza_remainder = pizza_area % 10

# Print the pizza remainder
print (f"If the pizza was cut into 10 square inch slices, there would be {pizza_remainder:.3f} square inches left for Sammi")

