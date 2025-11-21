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
