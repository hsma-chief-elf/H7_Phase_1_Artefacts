import random

# Create a dictionary to store scores from each round
scoreboard_dict = {}
# Set the round number to 0
round = 0

# Keep doing this infinitely
while True:
  # Start of the round
  # Print welcome messages
  print ("Welcome to the Number Guessing Game")
  print ("I'm thinking of a number.")

  # Add 1 to the round number
  round += 1

  # Set the score to 1000 at the start of the round
  score = 1000

  # Set number of guesses to 0
  num_guesses = 0

  # Set the Boolean that will indicate if they got it correct to False
  correct = False

  # Randomly generate the number to guess
  correct_num = random.randint(1,100)

  # Create a new list to store the user's guesses in this round
  list_of_guesses = []

  # Loop through and do this 10 times (10 guesses)
  for i in range(10):
    # Add 1 to the number of guesses
    num_guesses += 1

    # Keep doing this infinitely
    while True:
      # Ask the user to input their guess and cast as an integer
      guess = int(input(f"Guess {i+1}: "))

      # If the user hasn't already guessed this number, break out of this infinite loop
      if guess not in list_of_guesses:
        break

      # Otherwise print a message telling them they've already guessed this number
      print ("You already guessed that.  Try another.  Don't worry, we won't count that one.")
    
    # Add their guess to the list of guesses
    list_of_guesses.append(guess)

    # If the guess is correct
    if guess == correct_num:
      # Print a message confirming they are correct and how many guesses it took them
      print (f"Correct!  You got it in {num_guesses} guesses!")

      # Print a message with their score
      print (f"You scored {score}!")

      # Set the Boolean storing whether they got it correct to True
      correct = True

      # Break out of the for loop (end the round)
      break
    # Otherwise, if the guess is higher than the correct number
    elif guess > correct_num:
      # Tell them they're too high and decrement their score by 100
      print ("You're too high!")
      score -= 100
    # If they get here, then the guess must be too low
    else:
      # Tell them they're too low and decrement their score by 100
      print ("You're too low!")
      score -= 100

  # If they didn't guess correctly in this round, print a message telling them the answer
  if correct == False:
    print (f"Sorry you didn't get it.  The number was {correct_num}")

  # Store their score against the round number in the scoreboard dictionary
  scoreboard_dict[round] = score

  # Print out their guesses from this round
  print ("Your guesses this round were as follows :")
  print (list_of_guesses)

  # Ask if they want to play again
  again = input("Play again (y/n)? ")

  # If they answer no, break out of the outer while loop to stop playing any more rounds
  if again.lower() == "n":
    break

# Create a counter to add up their scores across the rounds they played, and start it at 0
sum_scores = 0

print ("SCOREBOARD")

# For each round and score in the scoreboard dictionary, print a message telling them their score
# for that round, and add that score to the sum of scores
for round, score in scoreboard_dict.items():
  print (f"Your score for round {round} was {score}")
  sum_scores += score

# Calculate the mean score as the sum of the scores divided by the number of rounds (length of the dictionary)
mean_score = sum_scores / len(scoreboard_dict)

# Print out their average score
print (f"Your average score was {mean_score:.2f}")

