for i in range(1,13):
    x=['']
    for j in range(1,10):
        x.extend([str(j*i)])

    #print x
    for n in x:
        print "%4s" % n,
    print
