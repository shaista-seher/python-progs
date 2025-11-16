# Python program to print all positive numbers in a list

# Example 1
list1 = [12, -7, 5, 64, -14]
print("Input:", list1)
print("Output:", end=" ")
for num in list1:
    if num > 0:
        print(num, end=" ")
print()

# Example 2
list2 = [12, 14, -95, 3]
print("Input:", list2)

# Storing positive numbers in a new list (as shown in example)
positive_numbers = []
for num in list2:
    if num > 0:
        positive_numbers.append(num)

print("Output:", positive_numbers)
