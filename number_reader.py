import time

answer=None

print "give me a number ",
answer=int(raw_input())
print type(answer)
while type(answer) != 'int':
    print "try again"
    answer=raw_input()


print "thank you ",
print answer
