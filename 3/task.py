
# 1-2 ///////////////////////////////////////////
class Item:
    # конструкор
    def __init__(self, count = 0, max_count = 100) -> None:
        self._count = count
        self._max_count = max_count
    
    # сетер
    def update_count(self, val):
        if val <= self._max_count:
            self._count = val
            return True
        else:
            return False
    
    # гетеры
    @property
    def count(self):
        return self._count
    
    @property
    def max_count(self):
        return self._max_count
    
    #------------------------------- операторы вычисления
    def __add__(self, num : int):
        if self.count + num <= self.max_count and self.count + num >= 0:
            self._count += num
        else:
            print("Your number increases the maximum size!")
        return self
    
    def __sub__(self, num : int):
        if self.count - num <= self.max_count and self.count - num >= 0:
            self._count -= num
        else:
            print("Your number increases the maximum size!")
        return self
    
    def __mul__(self, num : int):
        if self.count * num <= self.max_count and self.count * num >= 0:
            self._count *= num
        else:
            print("Your number increases the maximum size!")
        return self
    
    def __truediv__(self, num : int):
        if self.count / num <= self.max_count and self.count / num >= 0:
            self._count /= num
        else:
            print("Your number increases the maximum size!")
        return self
    
    # --------------------------------- сравнения
    def __eq__(self, num : int) -> bool:
        return self.count == num
    
    def __ne__(self, num : int) -> bool:
        return not (self == num)
    
    def __gt__(self, num : int) -> bool:
        return self.count > num
    
    def __ge__(self, num : int) -> bool:
        return self > num or self == num
    
    def __lt__(self, num : int) -> bool:
        return not (self >= num) 
    
    def __le__(self, num : int) -> bool:
        return self < num or self == num

# main1
myItem = Item(1, 100)
print (myItem.__dict__)
myItem._count = 5
print (myItem.__dict__)
myItem += 10
print (myItem.__dict__)
print (myItem > 5)

# 3-4 ////////////////////////////////////////////
# не понял
class Food():
    max_cell = 50
    amount_food = 0
    
    def __init__(self):
        print()

class Fruit():
    max_cell = 50
    amount_food = 0
    
    def __init__(self):
        print()
        
class AllFood:
    
    def __init__(self):
        print()

class Apple(Fruit):  
    def __init__(self):
        print("create apple")

class Pear(Fruit):
    def __init__(self):
        print("create pear")

class Potato(Food):
    def __init__(self):
        print("create Potato")
    
class Raspberry(Food):
    def __init__(self):
        print("create Raspberry")


# 5 /////////////////////////////////
class MyQueue:
    def __init__(self, base=[]):
        self._queue = base
    
    def __repr__(self):
        return str(self._queue)
    
    def get(self):
        return self._queue.copy()

    def push(self, elem):
        self._queue.append(elem)

    def pop(self):
        item = self._queue[0]
        self._queue = self._queue[1:-1]
        return item

# main 3
q1 = MyQueue([1,2,3,4,5])
print(q1.pop())
q1.push(100)
q2 = q1.get()
print(q2)
q2[2] = 1000
print(q1, q2)