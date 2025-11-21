# learn how to represent a person

class Person():
    
    def __init__(self, first_name, last_name):
        self.first_name = first_name[0].upper() + first_name[1:].lower()
        self.last_name = last_name[0].upper() + last_name[1:].lower()
        
    def __str__(self):
        # It would also have been correct to only format the
        # name here before you print them
        return '{} {}'.format(self.first_name, self.last_name)

person = Person("EmiLia", "GomEZ")
print(person)


# create a class to represent points in two dimensions

import math
class Point2D():
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def calculate_distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

# Solution testing
point1 = Point2D(3, 4)
point2 = Point2D(9, 5)
distance = point1.calculate_distance(point2)
print(distance)


# create a class to represent time of the day
class Time():
    
    def __init__(self, hour, minute, second):
        """
        Initialize a Time instance giving the hour, minute and second.
        """
        self.hour = hour
        self.minute = minute
        self.second = second
        
    def total_seconds(self):
        """
        Compute the total number of seconds since the start of the day.
        """
        return 3600 * self.hour + 60 * self.minute + self.second
    
    def __str__(self):
        """
        Create a string representation of this time.
        """
        s = ''
        if self.hour < 10:
            s += '0'
        s += str(self.hour)
        s += ':'
        if self.minute < 10:
            s += '0'
        s += str(self.minute)
        s += ':'
        if self.second < 10:
            s += '0'
        s += str(self.second)
        return s

# Solution testing
my_time = Time(9, 5, 7)
my_seconds = my_time.total_seconds()
print(my_seconds)
print(my_time)
