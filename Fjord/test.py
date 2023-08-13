# base = [[[0] for i in range(4)] for i in range(3)]
# # base = [[], [], []]
# a = [1, 2, 3]
# b = [1, 2, 3, 4]

# print(base)

# base[0].append(a)

# print(base)

# base[1].append(b)

# print(base)

# for L in base:
#     for l in L:
#         print(l)

import numpy as np

# list1 = np.array([[1,2,3], [4,5,6], [7,8,9]])
# list1 = np.lib.pad(list1, ((0,2),(0,2)), 'constant', constant_values=(0))
# print(list1)


# cd3=np.array([[1,2,3],[4,5,6],[4,5,6]])
# print(cd3)
# # cd3.resize((2,2),refcheck=False)
# cd2 = np.resize(cd3, (2,2))
# print(cd2)
# print(cd3)

# a = []
# shape = np.shape(list1)
# for i in range(len(shape)):
#     a.append((0,shape[i]))

# print(tuple(a))

a = [1, 2, 3]
print(np.sum(a[1:1]))