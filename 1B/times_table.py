
for i in range(1, 13):
  while True:
    print (f"Exploring the {i} times table")
    user_num = input("Input any whole number (q to move on to next one): ")

    if user_num.lower() == "q":
      break
    else:
      user_num = int(user_num)

    answer = i * user_num

    print (f"{i} times {user_num} is {answer}")

