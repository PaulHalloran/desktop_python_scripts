import sys, os, getopt

def count_stuff(file_name,test1,test2,test3):
    with open(file_name,"r") as inp:
        all_words=[]
        line_count=0
        for line in inp:
            line_count += +1
            [all_words.append(n) for n in (line.split(' '))]
    out=[]
    if test1 != -1: out.append(line_count)
    if test2 != -1: out.append(len(all_words))
    if test3 != -1: out.append("and words")  
    return out

input=[]

for arg in sys.argv[1:]:
    input.append(arg)

file_name=input[-1]

input2=' '.join(input[0:-1])

test1=test2=test3=-1

test1=input2.find("-l")
test2=input2.find("-w")
test3=input2.find("-c")

print count_stuff(file_name,test1,test2,test3)
