print "give me a year",
year=float(raw_input())

if year > 0:
    print "number is positive"

if year%4 == 0 or year%100 ==0:
    print "leap year"
