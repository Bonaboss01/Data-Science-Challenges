import datetime as dt
time_of_year = dt.datetime(year, month, day, hour, minutes, seconds)

time_to_format = dt.datetime(2020, 1, 25, 15, 25, 13)

print(time_to_format.strftime("%Y/%m/%d %H:%M:%S"))

  
