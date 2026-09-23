# Task 1

# Create the dictionary of Atari 800XL specs
atari_800xl_dict = {"release year":1983, "ram":64, "hres":320, "vres":190, "colours":256, "cpu speed":1.77, "graphics chip":"ANTIC"}

# Print the messages using information from the dictionary
print (f"The Atari 800XL was released in {atari_800xl_dict['release year']}")
print (f"Its graphics chip {atari_800xl_dict['graphics chip']} produced a resolution of {atari_800xl_dict['hres']} x {atari_800xl_dict['vres']} with {atari_800xl_dict['colours']} colours")
print (f"It had {atari_800xl_dict['ram']}kb of memory and a processor that ran at {atari_800xl_dict['cpu speed']}Mhz")

# Task 2

# Create the dictionary of student numbers
hsma_student_num_dict = {1:6, 2:26, 3:52, 4:80, 5:113, 6:178}

# Keep doing this infinitely
while True:
  # Ask the user to input a cohort number or q to quit
  selected_cohort = input("Enter a cohort number (or q to quit): ")

  # If they select to quit, break from the while loop
  if selected_cohort.lower() == "q":
    break
  else:
    # Otherwise convert the input to an integer
    selected_cohort = int(selected_cohort)

    # Attempt to lookup the index in the dictionary, returning 'cohort not found' if it doesn't exist
    num_students = hsma_student_num_dict.get(selected_cohort, "cohort not found")

    # Print an appropriate message depending on if the cohort was found or not
    if num_students == "cohort not found":
      print ("No such cohort.  Please try again.")
    else:
      print (f"In HSMA {selected_cohort} there were {num_students} students")

# Task 3

# Create dictionary of items and usage locations
mi_dict = {"pot":"circus tent", "red herring":"troll bridge", "shovel":"forest", "rubber chicken with pulley in the middle":"hook island","gopher repellent":"melee island jail"}

# Keep doing this infinitely
while True:
  # Ask the user to input their search term and convert to lowercase
  user_search = (input("Arrr enter your search term matey (or type 'loom' to quit): ")).lower()

  # If the user inputs loom, break from the while loop
  if user_search == "loom":
    break
  else:
    # Otherwise, set up two booleans to record if a match found in keys or values
    match_in_keys = False
    match_in_values = False
    
    # Loop through each key value pair in the dictionary.  If the search term is found in the string of the key, print a message
    # advising them to use the item at the location stored in the value, and set the boolean to True.  Also go "arrr".
    for key, value in mi_dict.items():
      if user_search in key:
        print (f"I think you need to use it at {value}.  Arrr.")
        match_in_keys = True
    
    # If a match wasn't found in the keys, print an appropriate message and try searching the values.
    if match_in_keys == False:
      print ("Didn't find any matches in the items list.  Checking location list.")
    
      for key, value in mi_dict.items():
        if user_search in value:
          print (f"I think you need to use the {key}.  Arrr.")
          match_in_values = True
    
      # If nothing could be found in the values either, print a message and advise they ask you about Loom.
      if match_in_values == False:
        print ("Sorry, I couldn't find anything.  Ask me about Loom.")
    
# Print a message telling them about Loom.
print ("You mean the latest masterpiece of fantasy storytelling from Lucasfilm's™ Brian Moriarty™? Why, it's an extraordinary adventure with an interface of magic, stunning, high-resolution, 3D landscapes, sophisticated score and musical effects. Not to mention the detailed animation and special effects, elegant point 'n' click control of characters, objects, and magic spells. Beat the rush! Go out and buy Loom™ today!")