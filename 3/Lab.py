class MyMatrix():
    def __init__(self, file_1, file_2):
        self.mat1 = MyMatrix.readMatrix(file_1)
        self.mat2 = MyMatrix.readMatrix(file_2)
        
        if (MyMatrix.checkMatrix(self.mat1, self.mat2) == False):
            self.mat1 = [[]]
            self.mat2 = [[]]
            ValueError("Mатрицы должны быть одинакового размера")
    
    @classmethod
    def checkMatrix(cls, mat1 , mat2):
        if (len(mat1) != len(mat2)):
            return False
        if (len(mat1[0]) != len(mat2[0])):
            return False
        return True

    @classmethod
    def readMatrix(cls, file):
        mat = []
        fileRead = open(file)
        for line in fileRead:
            mat.append(line.split())
        fileRead.close()
        return mat

    def printMatrix(self, mat):
        for i in range(len(mat)):
            print(str(mat[i]))
    
    def plusMatrix(self):
        res = []
        for i in range(len(self.mat1)):
            line = []
            for j in range(len(self.mat1[0])):
                line.append(int(self.mat1[i][j]) + int(self.mat2[i][j]))
            res.append(line)
        self.printMatrix(res)
    
    def minusMatrix(self):
        res = []
        for i in range(len(self.mat1)):
            line = []
            for j in range(len(self.mat1[0])):
                line.append(int(self.mat1[i][j]) - int(self.mat2[i][j]))
            res.append(line)
        self.printMatrix(res)
            

class Pupa:
    def __init__(self, money = 0):
        self._money = money
        print("create Pupa")
    
    def take_salary(self, money):
        self._money += money
    
    def update_money(self, money):
        self._money = money
    
    def do_work(self, file_name_1, file_name_2):
        mat1 = MyMatrix(file_name_1, file_name_2)
        mat1.minusMatrix()
        # print(mat1.__dict__)
        
    @property
    def money(self):
        return self.money
        
class Lupa:
    def __init__(self, money = 0):
        self._money = money
        print("create Lupa")
    
    def update_money(self, money):
        self._money = money
        
    @property
    def money(self):
        return self.money
    
    def take_salary(self, money):
        print(self.__dict__)
        self._money += money
    
    def do_work(self, file_name_1, file_name_2):
        mat1 = MyMatrix(file_name_1, file_name_2)
        mat1.plusMatrix()
        # print(mat1.__dict__)
          
class Accountant:
    def give_salary(self, worker, money):
        worker.take_salary(money)
        

# //////
acc = Accountant()
lupa = Lupa()
pupa = Pupa()

acc.give_salary(lupa, 100)
acc.give_salary(pupa, 500)

lupa.do_work("3/t1.txt", "3/t2.txt")