import random
import string

def isPolendrom(text: string):
    reverseText = "".join(reversed(text))
    if (reverseText == text):
        print("Polendrom")
    else:
        print("UnPolendrom")

def searchLongWord(text):
    index = 0;
    maxLen = 0;
    if len(text) == 0:
        print("Empty myList")
        return
    for i in range(len(text)):
        if len(text[i]) > maxLen:
            index = i;
            maxLen = len(text[i]);
    print(text[index])
    
def amountColChetNeChet(myList):
    chet = 0;
    noChet = 0;
    print(myList)
    for i in range(len(myList)):
        if myList[i] % 2 == 0:
            chet += 1
        else:
            noChet += 1    
    print('chet = ' + str(chet) + "\nnoChet = " + str(noChet))

def changeArray(myList, text):
    for e in myList.items():
        print(e)
        text = text.replace(e[0], e[1])
    print(text)

def generateProgr(step, elem, amount):
    for i in range(amount+1):
        yield elem
        elem *= step

# main
#1
text = input("Your text: ")
isPolendrom(text)

#2
myList = ["самое", "длинное", "слвоо"];
searchLongWord(myList)

#3
numbers = [random.randint(0, 100) for i in range(10)]
amountColChetNeChet(numbers)

# 4
myList = {"bad": "good", "dog":"cat"}
text = "a bad person is better than a good one, and a dog is the enemy of a cat"
changeArray(myList, text)

# 5
def fib(n : int) -> int:
    if(n == 0): return 0
    if(n == 1): return 1
    return fib(n-1) + fib(n-2)
print(fib(9))

# 6
with open("text.txt", "r") as file:
    lines, words, symbols = 0, 0, 0
    file_content = file.readlines()
    lines = len(file_content)
    for line in file_content:
        symbols += len(line.split(' ')) - 1
        for word in line.split(' '):
            words += 1
            for symbol in word:
                symbols += 1
print(f"Count of lines: {lines}\nCount of words: {words}\nCount of symbols: {symbols}")

# 7
amount = 5
elem = 2
step = 2

for i in generateProgr(step, elem, amount):
    print(elem, end=' ')
    elem *= step