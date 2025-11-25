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


# function to count smaller values

test_values = [5, 3, 7, 8, 1, 10]
test_query_values = [7, 2, 11, 1, 10]
# Expected answer: [3, 1, 6, 0, 5]
import bisect

# O(N log(N) + Q log(N)) where N = len(values) and Q = len(query_values)
def count_smaller(values, query_values):
    sorted_values = sorted(values)                         # N log(N)
    count = []                                             # 1
    for query_value in query_values:                       # Q
        i = bisect.bisect_left(sorted_values, query_value) # Q log(N)
        count.append(i)                                    # 1
    return count

print(count_smaller(test_values, test_query_values))
