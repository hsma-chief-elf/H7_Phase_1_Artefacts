
print ("Welcome to the number multiplier!")
print ("Enter q at any point to stop")

while True:
  num_1 = input("Give me the first number: ")
  if num_1.lower() == "q":
    break
  else:
    num_1 = float(num_1)
    
  num_2 = input("Give me the second number: ")
  if num_2.lower() == "q":
    break
  else:
    num_2 = float(num_2)
  
  product = num_1 * num_2
  print (f"The product is {product:.2f}")

print ("Cheerio!")

