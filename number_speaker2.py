print 'give me a number: ',
answer=raw_input()

numbers=['zero','one','two','three','four','five','six','seven','eight','nine']

answer2=[]
answer2.extend(answer)
while answer2 != []:
    print numbers[int(answer2[0])],
    answer2[0:1] = []    

