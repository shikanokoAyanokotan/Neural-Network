from numpy import random



# random.randint(N): Generate a random integer in range [0, N]
rInt = random.randint(100)
print(rInt)


# random.rand(): Generate a random float number in range [0, 1]
rFloat = random.rand()
print(rFloat)


# Generate a 1-D array containing 5 random integers from 0 to 100:
r1DArray = random.randint(100, size=(5))
print(r1DArray)


# Generate a 2-D array, each row containing 5 random integers 
# from 0 to 100:
r2DArray = random.randint(100, size=(3, 5))
print(r2DArray)


# Generate a 1-D array containing 5 random floats
r1DfloatArray = random.rand(5)
print(r1DfloatArray)


# Generate a 2-D array containing
r2DfloatArray = random.rand(3, 5)
print(r2DfloatArray)


# Return one of the values in an array
randomChoice = random.choice([3, 5, 7, 9])
print(randomChoice)


# Generate a 2-D array that consists of the values in the array 
# parameter (3, 5, 7, 9)
random2DarrayChoice = random.choice([3, 5, 7, 9], size=(3, 5))
print(random2DarrayChoice)
