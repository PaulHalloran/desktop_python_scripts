def input():
    print "give me a number",

    answer=None

    while type(answer) != 'int':
        answer=raw_input()
        if answer.isdigit():
            print "thank you"
            return answer
        else:
            print "try again",
            answer=raw_input()

print input()
