def readSequence( fn ):
    sequence = ""
    with open(fn) as f:
        for line in f:
            elements = line.split()
            sequence += "".join(elements[1:])
    return sequence

def getGene( sequence, begin, end ):
    return sequence[begin-1:end]

def findAll( sequence, pattern ):
    pos = []
    next = sequence.find(pattern)
    while next != -1:
        pos.append(next)
        next = sequence.find(pattern, next+1)
    return pos

def findDinuc( sequence):
    dinucs = ["aa", "tt", "gg", "cc"]
    result = {}
    for p in dinucs:
        positions = findAll( sequence, p )
        result[p] = positions
    return result

def countDinuc(dinucs):
    sum = 0
    for n in dinucs.values():
        sum += len(n)
    return sum

def saveDinuc( fn, dinucs ):
    with open(fn,"w") as f:
        for nn in dinucs.keys():
            for pos in dinucs[nn]:
                line = nn + ", " + str(pos) + "\n"
                f.write(line)


if __name__ == "__main__":
    dna = readSequence( "rIIAB.txt" )
    rIIB = getGene( dna, 1461, 2399 )
    dinuc = findDinuc( rIIB )
    print( f"{countDinuc(dinuc)} found" )
    saveDinuc( "rIIB.csv", dinuc )
