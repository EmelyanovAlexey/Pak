import argparse

def readDate(fileRead):
    mat1 = []
    mat2 = []
    isMat2 = False
    for line in fileRead:
        if (len(line) != 1 and isMat2 == False):
            mat1.append(line.split())
        if (len(line) != 1 and isMat2):
            mat2.append(line.split())
        if (len(line) == 1):
            isMat2 = True
    return mat1, mat2

def printArr(arr):
    for i in arr:
        print(i)

def checkMatrix(mat1, mat2):
    if (len(mat1[0]) != len(mat2)) or (len(mat1) != len(mat2[0])):
        return False
    return True

def multyMatrix(mat1, mat2, fileWrite):
    res = []
    column = len(mat1[0])
    row = len(mat1)

    for i in range(row):
        myRow = []
        for j in range(row):
            sum = 0
            for x in range(column):
                sum += int(mat1[i][x]) * int(mat2[x][j])
            myRow.append(sum)
        res.append(myRow)
    print(res)
    with open(fileWrite, "w") as f:
        for i in res:
            for j in i:
                f.write(str(j) + " ")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fileRead", metavar="name file read matrix", type=int,
        nargs=1, help="file read matrix")
    parser.add_argument("writeRead", metavar="name file write matrix", type=int,
        nargs=1, help="file write matrix")
    args = parser.parse_args()

    fileRead = open(args.fileRead[0])
    mat1, mat2 = readDate(fileRead)
    printArr(mat1)
    printArr(mat2)
    if (checkMatrix(mat1, mat2)):
        multyMatrix(mat1, mat2, args.writeRead[0])
    else:
        print("! Error\ndifrent len martrix")

main()