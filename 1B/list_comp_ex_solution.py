# Task 1

# Create list of Murder She Wrote characters
msr_chars = ["Jessica Fletcher", "Amos Tupper", "Seth Hazlitt", "Grady Fletcher", "Dennis Stanton", "Andy Broom"]

# Use list comprehension to generate new list with all names converted to lowercase
lower_chars = [name.lower() for name in msr_chars]

# Print the list
print (lower_chars)

# Task 2

# Create list of floats
list_of_floats = [1.279, 2.167, 2.467, 3.912, 2.87]

# Use list comprehension to generate new list containing only those numbers from list_a which
# are both less than 3 and whose value squared is more than 5.
filtered_list_of_floats = [num for num in list_of_floats if num**2 > 5 and num<3]

print (filtered_list_of_floats)

# Task 3

# Create list of Murder She Wrote characters
msr_chars = ["Jessica Fletcher", "Amos Tupper", "Seth Hazlitt", "Grady Fletcher", "Dennis Stanton", "Andy Broom"]

# Use list comprehension to generate new list containing only those names that don't start with a vowel
consonant_only_chars = [name for name in msr_chars if name[0].lower() not in ["a","e","i","o","u"]]

print (consonant_only_chars)

# Task 4

# List of season 12 first 12 episode IMDB scores
season_12_part_1 = [7.1, 7.3, 7.1, 7.2, 7.0, 7.4, 6.8, 7.0, 7.6, 6.5, 7.1, 7.5]

# List of season 12 final 11 episode IMDB scores
season_12_part_2 = [7.1, 7.3, 6.9, 7.5, 7.2, 7.1, 6.8, 7.1, 7.2, 7.2, 7.2]

# Calculate the mean IMDB score of the first 12 episodes
part_1_mean = sum(season_12_part_1)/len(season_12_part_1)

# Use list comprehension to create a new list that contains only those scores from 
# the second half of season 12 that are at least as high as the average score from
# the first 12 episodes
better_than_p1_average_list = [score for score in season_12_part_2 if score>=part_1_mean]

# Print the list of scores that are better than the first half average
print (better_than_p1_average_list)

# Calculate the percentage of second half episodes that are at least as well rates as the first half average
perc_better_than_p1_average = len(better_than_p1_average_list) / len(season_12_part_2)

# Print a message specifying the percentage calculated above
print (f"{perc_better_than_p1_average*100:.2f}% of final episodes were better than the first half of the season average")

