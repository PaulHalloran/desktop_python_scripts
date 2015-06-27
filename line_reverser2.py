def rev(lst):
    if len(lst) == 0:
        return
    else:
        rev(lst[1:])
        print lst[0]

rev(['hello','paul','you','fool'])
