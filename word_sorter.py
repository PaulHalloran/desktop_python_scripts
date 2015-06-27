answer=[None]

print 'give me a word ',

while answer[-1] != '':
    answer.append(raw_input())
    print 'and another',

print sorted(answer[1:-1])
