from typing import List, Dict

def readSequence(fn : str ) -> str :
    sequence = ""
    with open(fn) as f:
        for line in f:
            elements = line.split()
            sequence += "".join(elements[1:])
    return sequence

def getGene( sequence : str , begin:int, end:int ) -> str:
    return sequence[begin-1:end]

def findAll( sequence : str, pattern : str) -> List[int]:
    pos = []
    next = sequence.find(pattern)
    while next != -1:
        pos.append(next)
        next = sequence.find(pattern, next+1)
    return pos

def findDinuc( sequence: str) -> Dict[str, List[int]]:
    dinucs = ["aa", "tt", "gg", "cc"]
    result = {}
    for p in dinucs:
        positions = findAll( sequence, p )
        result[p] = positions
    return result

def countDinuc(dinucs: Dict[str, List[int]]) -> int:
    sum = 0
    for n in dinucs.values():
        sum += len(n)
    return sum

def saveDinuc( fn:str, dinucs:  Dict[str, List[int]] ) -> None:
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
