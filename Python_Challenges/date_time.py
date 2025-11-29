# Investigating with different timedate packages
import datetime as dt
time_of_year = dt.datetime(year, month, day, hour, minutes, seconds)

time_to_format = dt.datetime(2020, 1, 25, 15, 25, 13)

print(time_to_format.strftime("%Y/%m/%d %H:%M:%S"))

time_str = "12:30 2019-05-19"

date_object = dt.datetime.strptime(time_str, "%H:%M %Y-%m-%d")

import pytz

my_timezone_name = 'America/New_York' 
time_now = dt.datetime.now(pytz.timezone(my_timezone_name))

time_now = dt.datetime.now()

# Birthday for someone born on 19th of May
time_birthday = dt.datetime(time_now.year, 5, 19)
time_to_birthday = time_birthday - time_now
print(time_to_birthday)
