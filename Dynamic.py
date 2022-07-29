import matplotlib.pyplot as plt

capacity = 17000
initCharge = 5000
priceSchedule = [1.5, 1.5, 1.5, 1.5,
                 1.5, 1.5, 2, 3, 5,
                 5, 5, 4.5, 3, 3, 3,
                 3, 4.5, 5, 7, 9,
                 11, 12, 8, 4]
loadSchedule = [480, 320, 320, 360, 360,
                360, 420, 920, 1200, 720,
                680, 720, 800, 820, 960,
                1200, 1380, 1380, 1520,
                1800, 1920, 1920, 1640,
                1020]
constantLoad = 350
targetCharge = 5000
chargeValue = [-4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000]


def forward_calculating(dictionary):
    index_of_best_value = 0
    counter = 9
    best_value = float('-inf')
    for k in range(2, 25):
        iteration_value, counter = counter, 0
        for i in range(iteration_value):
            for j in chargeValue:
                try:
                    if (0 <= (dictionary['g' + str(k - 1) + str(i)] -
                              (loadSchedule[k - 1] + constantLoad) + j) <= capacity): # Отсекаем все что не попадает
                        # в интервал заряда (0-Макс)
                        dictionary['f' + str(k) + str(counter)] = \
                            (dictionary['f' + str(k - 1) + str(i)]) + \
                            (priceSchedule[k - 1] * (j * -0.001))

                        dictionary['g' + str(k) + str(counter)] = \
                            (dictionary['g' + str(k - 1) + str(i)] -
                             (loadSchedule[k - 1] + constantLoad) + j)

                        dictionary['x' + str(k) + str(counter)] = j
                        if k == 24:

                            if dictionary['g' + str(k) + str(counter)] < targetCharge:
                                removing_objectionable(counter, dictionary, k)
                            elif ((dictionary['g' + str(k) + str(counter)] >= targetCharge) & (
                                    dictionary[
                                        'f' + str(k) + str(counter)] > best_value)):
                                best_value = dictionary['f' + str(k) + str(counter)]
                                index_of_best_value = counter
                        counter += 1
                except KeyError:
                    continue
        for m in range(counter):
            for l in range(counter):
                try:
                    this_capacity = dictionary['g' + str(k) + str(m)]
                    previous_capacity = dictionary['g' + str(k) + str(l)]
                    this_function_value = dictionary['f' + str(k) + str(m)]
                    previous_function_value = dictionary['f' + str(k) + str(l)]
                    if (this_capacity == previous_capacity) & \
                            (this_function_value >= previous_function_value) & \
                            (m != l):
                        removing_objectionable(l, dictionary, k)
                except KeyError:
                    continue
    return dictionary, index_of_best_value, best_value


def removing_objectionable(counter, dictionary, k):
    del dictionary['x' + str(k) + str(counter)]
    del dictionary['g' + str(k) + str(counter)]
    del dictionary['f' + str(k) + str(counter)]


def reverse_folding(dict_calc_final, best_value_index):
    final_mass = [k for k in range(24)]
    num_of_best_function = best_value_index
    final_mass[0] = initCharge
    final_mass[23] = dict_calc_final['g' + str(24) + str(best_value_index)]
    dict_calc_len = len(dict_calc_final)
    for k in range(22):
        previous_value = dict_calc_final['g' + str(24 - k) + str(num_of_best_function)] - \
                         dict_calc_final['x' + str(24 - k) + str(num_of_best_function)] + \
                         (loadSchedule[24 - k - 1] + constantLoad)
        for i in range(dict_calc_len):
            try:
                if dict_calc_final['g' + str(24 - k - 1) + str(i)] == previous_value:
                    final_mass[24 - k - 2] = previous_value
                    num_of_best_function = i
            except KeyError:
                continue
    return final_mass


table = {}

for i in range(len(chargeValue)):
    table['x' + str(i)] = chargeValue[i]
    table['f1' + str(i)] = priceSchedule[0] * (chargeValue[i] * -0.001)
    table['g1' + str(i)] = targetCharge - (loadSchedule[0] + constantLoad) + chargeValue[i]
    print(str(table['x' + str(i)]) + " " + str(table['f1' + str(i)]) + " " + str(table['g1' + str(i)]))

print(table)



dict_24, index_of_best_value, best_value = forward_calculating(table)


print("Значение наилучшей целевой функции: ", best_value)

final_mass = reverse_folding(dict_24, index_of_best_value)

print("Значения заряда батареи для оптимальных торговых операций: ")
print(final_mass)

without_sale_charge = []

for i in range(24):
    initCharge -= (constantLoad + loadSchedule[i])
    if initCharge <= 0:
        initCharge = 0
        without_sale_charge.append(initCharge)
    else:
        without_sale_charge.append(initCharge)

time = [i for i in range(24)]
plt.plot(time, without_sale_charge, label="Без торгов")
plt.plot(time, final_mass, label="С учетом торгов")

plt.grid()
plt.xlabel('Время (час)')
plt.ylabel('Уровень заряда (Втч)')
plt.legend()

plt.show()
