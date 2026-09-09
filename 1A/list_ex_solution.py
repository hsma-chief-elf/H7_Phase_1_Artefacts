# Task 1

# Create a new list of MSW imdb scores for first 10 episodes of season 1
list_of_s1_imdb_scores = [7.9, 7.4, 7.3, 7.5, 7.3, 7.3, 7.5, 7.5, 7.3, 7.5]

# Add scores for 11th and 12th episodes
list_of_s1_imdb_scores.append(7.5)
list_of_s1_imdb_scores.append(7.1)

# Change the score of episode 2 to 8.1
list_of_s1_imdb_scores[1] = 8.1

# Print the list
print (list_of_s1_imdb_scores)

# Print the score for episode 3
print (f"Episode 3 score: {list_of_s1_imdb_scores[2]}")

# Print the list of scores for episodes 2 to 6
print (f"Episodes 2 to 6 scores: {list_of_s1_imdb_scores[1:6]}")

# Print the list of scores for the first 5 episodes
print (f"First 5 episode scores: {list_of_s1_imdb_scores[:5]}")

# Print the score for the last episode in the list
print (f"Episode 12 score: {list_of_s1_imdb_scores[-1]}")

# Remove 5th episode score from list
list_of_s1_imdb_scores.pop(4)

# Print a confirmation of how many episodes are now in the list
print (f"There are now {len(list_of_s1_imdb_scores)} episodes in the list")

# Task 2

# Create list of murder weapons for first 10 episodes
list_of_murder_weapons = ["shotgun", "handgun", "handgun", "metal urn", "iron gate", "pearl necklace", "driverless car", "bookend", "handgun", "knife"]

# Check if at least one of the murder weapons used an iron or wooden gate
if "iron gate" in list_of_murder_weapons or "wooden gate" in list_of_murder_weapons:
  print ("At least one of the murders involved a gate")
else:
  print ("There are no recorded gate related murders in the first 10 episodes")

# Check that a frozen yellowtail fish isn't already in the list of murder weapons, and add it if that's the case
if "frozen yellowtail fish" not in list_of_murder_weapons:
  list_of_murder_weapons.append("frozen yellowtail fish")

# Print the list of murder weapons
print (list_of_murder_weapons)

# Count how many handguns and shotguns there were, then print the total
number_of_handguns = list_of_murder_weapons.count("handgun")
number_of_shotguns = list_of_murder_weapons.count("shotgun")
print (f"A total of {number_of_handguns+number_of_shotguns} guns were used as murder weapons")

# Create new list of murder weapons for first 10 episodes of season 10
s10_list_of_murder_weapons = ["poison dart", "handgun", "wooden stake", "rigged radio", "letter opener", "handgun", "curtain cord", "handgun", "handgun", "scalpel"]

# Extend the original list with this new list
list_of_murder_weapons.extend(s10_list_of_murder_weapons)

# Print combined list of murder weapons, as well as how many guns feature
print (list_of_murder_weapons)
number_of_handguns = list_of_murder_weapons.count("handgun")
number_of_shotguns = list_of_murder_weapons.count("shotgun")
print (f"A total of {number_of_handguns+number_of_shotguns} guns were used as murder weapons across both sets of episodes")

# Task 3

# Ask the user for their name
name = input("What is your name? ")

# If there is no space in the name, ask for their surname and then add that to the name
if " " not in name:
  surname = input("What is your surname? ")
  name = name + " " + surname

# Greet them with their name
print (f"Hello {name}")

# Check if the name contains all five vowels (in any case) and if it does, print an appropriate message
if "a" in name.lower() and "e" in name.lower() and "i" in name.lower() and "o" in name.lower() and "u" in name.lower():
  print ("Hey, your name uses all five vowels!")

