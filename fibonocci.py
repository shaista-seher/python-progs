# Python program to print Fibonacci sequence

# Number of terms you want
n = int(input("Enter how many Fibonacci numbers you want: "))

# First two Fibonacci numbers
a, b = 0, 1

print("Fibonacci sequence:")

# Loop to generate Fibonacci numbers
for i in range(n):
    print(a, end=" ")
    a, b = b, a + b
