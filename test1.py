#!/usr/local/sci/bin/python2.7
#this makes the program executable, so you can run it by just typing 'test1.py' on command line

'''
This is some test code
'''
#the above is a diff way of labling code

print "what is your name? ",
#comma means that the following stuff comes on the same line
name=raw_input()
if name=='':
    #note that need the colon
    print "no name supplied"
    name="annon"
elif name == "Bob":
    # the colon is essentiall in place of the 'then' keyword
    print "very good answer"
else:
    print "good"
print "hello",name
