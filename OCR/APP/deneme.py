with open("requirement.txt","r") as file:
    f = file.read()
file.close()

tf_list = []

for i in f.split("=="):
    tf_list.append(i[:-1])

print(tf_list)
