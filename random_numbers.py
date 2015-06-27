import math
import random

def random_numbers(n):
    count=0
    while count < n:
        print random.random()
        print math.sin((random.random()))
        count += +1

random_numbers(5)
