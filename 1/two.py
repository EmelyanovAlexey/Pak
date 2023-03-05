import argparse
import random

def generateList(N: int):
    list = []
    for i in range(N):
        list.append(random.randint(0, 100))
    return list

def sortList(list):
    for i in range(len(list)-1):
        for i in range(len(list)-i-1):
            if (list[i] > list[i+1]):
                list[i], list[i+1] = list[i+1], list[i]
    return list

def myPrint(list):
    for i in range(len(list)):
        print(list[i], end=' ')
    print('')

def PrintPasTriangle(rowsLen: int):
    allList = [[]]
    line = []
    strLen = 3
    for i in range(1, rowsLen + 1):
        line.insert(0, 1)
        for j in range(1, len(line) - 1):
            line[j] += line[j + 1]
        allList.append("".join(str(line)))
        strLen = len("".join(str(line)))
    for i in range(1, rowsLen + 1):
        print(allList[i].center(strLen))

def sort2(L):
    newList = []
    for i in range(0,len(L)):
        
        newList.append(L[i])
        while(len(newList) >= 2 and L[i] > newList[-2]):
            newList.pop(-2)

        #print(newList[0])
        if( i > 2 and L[i-3] == newList[0]):
            newList.pop(0)
        if(i >= 2):
            print(newList[0], end=' ')

# main
parser = argparse.ArgumentParser()
parser.add_argument("N", metavar="lenList", type=int,
                    nargs=1, help="lenght my list")
args = parser.parse_args()


list = []
list = generateList(args.N[0])
myPrint(list)
list = sortList(list)
myPrint(list)


list = []
PrintPasTriangle(args.N[0])

list = []
list = generateList(args.N[0])
myPrint(list)
sort2(list)
