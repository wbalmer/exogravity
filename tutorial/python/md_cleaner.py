import sys

BALISE = "``` {.python linerange="

args = sys.argv
filename = args[1]

f = open(filename, "r")
lines = f.readlines()
f.close()

f = open(filename, "w")
k = 0
while (k<len(lines)):
    line = lines[k]
    splitted = line.split(BALISE)
    if len(splitted)<2:
        f.write(line)
        k = k+1
    else:
        f.write("``` code\n")
        k = k+1
        rows = splitted[1].split('}')[0].replace('"', '').split('-')
        l1, l2 = (int(rows[0]), int(rows[1]))
        k = k+(l1-1)
        for j in range(l2-l1+1):
            f.write(lines[k])
            k = k+1
        end = False
        while not(end):
            if lines[k] == "```\n":
                end = True
            k = k+1
        f.write('```\n')
            
f.close()

