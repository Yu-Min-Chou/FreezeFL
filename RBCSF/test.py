
# class Client:
#     def __init__(self):
#         self._a = 10

#     @property
#     def a(self):
#         self._a

#     def where_a(self):
#         print(hex(id(self._a)))
#         a = self._a
#         print(hex(id(client.a)))

# client = Client()
# a = client.a
# print(hex(id(client.a)))
# print(hex(id(a)))
# client.where_a()

a = 10
b = []

b.append(a)
print(hex(id(a)))
print(hex(id(b[0])))

a = 4
print(b[0])

print(hex(id(a)))
print(hex(id(b[0])))