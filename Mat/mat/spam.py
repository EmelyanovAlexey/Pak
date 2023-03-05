import string
import math

# -------------------------- MAIN
# константы на количество файлов
CONST_AMOUNT_SPAM = 2
CONST_AMOUNT_NO_SPAM = 2
CONST_AMOUNT_SMS = 1

# сначало, мы должны обучить программу, предоставив программе данные
# какие слова спам а какие не спам, а далее будем проверять сообщения

# сначало пичкаем spam для обучения

spamList = []

for i in range(CONST_AMOUNT_SPAM):
    fileName = "spam/spam" + str(i+1) + ".txt"
    fileRead = open(fileName, encoding="UTF-8")
    for line in fileRead:
        # считываю строку удаляя все знаки , и привожу к нижнему регистру
        words = line.translate(str.maketrans('', '', string.punctuation)).lower()
        arrWords = words.split()
        for wordItem in arrWords:
            # если слова нету в списке то добавляю
            if (spamList.count(wordItem) == 0):
                spamList.append(wordItem)
    fileRead.close()

print(spamList)
# print(str(len(spamList)))

# Далее пичкаем noSpam для обучения

# print('-----------------------')

noSpamList = []

for i in range(CONST_AMOUNT_NO_SPAM):
    fileName = "noSpam/noSpam" + str(i+1) + ".txt"
    fileRead = open(fileName, encoding="UTF-8")
    for line in fileRead:
        # считываю строку удаляя все знаки , и привожу к нижнему регистру
        words = line.translate(str.maketrans('', '', string.punctuation)).lower()
        arrWords = words.split()
        for wordItem in arrWords:
            if (noSpamList.count(wordItem) == 0):
                noSpamList.append(wordItem)
    fileRead.close()

print(noSpamList)
# print(str(len(noSpamList)))

# d в итоге у нас сохранены слова спама и не спама при чем, мы их сохраняем
# в едиснтвенном экземпляре. и начинаем считывать письма

# Далее проверяем письма

# print('-----------------------')

for i in range(CONST_AMOUNT_SMS):
    smsList = []
    fileName = "sms/sms" + str(i+1) + ".txt"
    
    # считываем письмо
    fileRead = open(fileName, encoding="UTF-8")
    for line in fileRead:
        # считываю строку удаляя все знаки , и привожу к нижнему регистру
        words = line.translate(str.maketrans('', '', string.punctuation)).lower()
        arrWords = words.split()
        for wordItem in arrWords:
            smsList.append(wordItem)
    fileRead.close()
    
    print(smsList)
    # print(str(len(smsList)))
    
    # итак теперь самое главное, мы считали сообщение и начинаем проверять это письмо спам или нет
    # для этого пользуемся формулой баяса для каждого слова
    # a = количество повторов слов в сообщениие
    # b = количество повторов слов в spam/noSpam ( мы их отдельно считаем) всегда 1
    # c = количество слов в сообщении
    # z = количество слов в спам/ неспам
    
    # p = (a+b)/(c+z) чтобы убрать погрешности мы по Лапласу должны в числитель прибавит 1 а в знаменатель 2
    # тогда p = (a+b+1)/(c+z+2)
    # но если слова из сообщения не встретилось в спам/не спам то p = 1/2
    
    # теперь нам осталось сложить сумму log(p) всего сообщения для спам и не спам
    # и останется сравнить 118 строка
    
    # анализируем письмо
    multy_spam = 0
    multy_no_spam = 0
    
    for smsItem in smsList:
        p_spam = 0
        p_no_spam = 0;

        if (spamList.count(smsItem) > 0):
            p_spam = math.log((smsList.count(smsItem) + 2)/(len(smsList) + len(spamList) + 2))
            #print(smsItem + str(p_spam))
        else:
             p_spam = math.log(0.5)
                   
        if (noSpamList.count(smsItem) > 0):
            p_no_spam = math.log((smsList.count(smsItem) + 2)/(len(smsList) + len(noSpamList) + 2))
            #print(smsItem + str(p_no_spam))
        else:
            p_no_spam = math.log(0.5)
        
        # Оценка для категории «Спам»:
        multy_spam += p_spam
        
        # Оценка для категории «Не спам»:
        multy_no_spam += p_no_spam
        
    if (multy_spam > multy_no_spam):
        print('spam ' + str(multy_spam))
    else:
        print('no spam ' + str(multy_no_spam))


# в моей программе не хватает 1 погрешности ну Анисова приняла, вчера понял только как ее делать
# в общем когда считаешь сообщения там можешь сделать пометку от себя спам это или нет
# по итогу когда сообщения все считала и сравниваешь какие сообщения твой анализ определил верно
# ну и по формуле P all = (количество определенных верно) / (на все сообщения)