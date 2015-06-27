print 'give me a number: ',
answer=raw_input()

numbers=['zero','one','two','three','four','five','six','seven','eight','nine']

answer2=[None]
answer2.extend(answer)
for i in answer2[1:]:
    print numbers[int(i)],
