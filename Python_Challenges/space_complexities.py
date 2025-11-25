# an algorithm that finds out whether there are two values whose difference is equal to the target value.
test_values = [5, 7, 2, 8]
test_target1 = 3
# Expected answer: True

test_target2 = 4
# Expected answer: False
def find_pair_with_difference(values, target):
    value_set = set(values)
    for value2 in values:
        # value1 - value2 = target means that value1 = target + value2 
        value1 = target + value2
        if value1 != value2 and value1 in value_set:
            return True
    return False

print(find_pair_with_difference(test_values, test_target1))
print(find_pair_with_difference(test_values, test_target2))
