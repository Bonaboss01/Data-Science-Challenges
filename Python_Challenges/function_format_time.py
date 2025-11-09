# Write a function named format_time with two arguments:
# hour: An integer representing the hour.
# minute: An integer representing the minute.
# Implement the format_time() function so that it formats and outputs the provided hour and minute into a string with five characters as described above.
# Test your function by printing the result on the provided inputs. For example, you can test it on the first input by executing print(format_time(hour_1, minute_1)).

# Answer

# provided inputs
hour_1 = 10
minute_1 = 30 
# answer to this input: 10:30

hour_2 = 5
minute_2 = 7 
# answer to this input: 05:07

hour_3 = 16
minute_3 = 9 
# answer to this input: 16:09

def format_time(hour, minute):
    return (hour,":",minute)

print(format_time(hour_1, minute_1))
