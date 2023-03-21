ids = ['id1', 'id100', 'id2', 'id22', 'id3', 'id30']
ids2 = [1, 100, 2, 22, 3, 30]

sorted_ids = sorted(ids, key=lambda x: int(x[2:])) # Integer sort

print(ids.size)