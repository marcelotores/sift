# x = lambda a, b: a * b
#
# print(x = lambda a, b: a * b)

ids = ['id1', 'id100', 'id2', 'id22', 'id3', 'id30']
sorted_ids = sorted(ids, key=lambda x: int(x[2:])) # Integer sort
print(sorted_ids)