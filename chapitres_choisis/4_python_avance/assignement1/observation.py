from typing import List, Dict, Set

Entries = List[Dict[str,str]]

def readData( fn: str)->Entries:
    res = []
    with open(fn,'r') as f:
        for line in f:
            tokens = line.rstrip().split()
            sp = tokens[0][0:3] + "_" + tokens[1][0:3]
            row = { "species": sp, "location": tokens[2], "year": tokens[3] }
            res.append(row)
    return res

def saveCSV( entries: Entries, out_file: str)->None:
    with open( out_file, "w" ) as f:
        for r in entries:
            line = " ".join([ r['species'], r['location'], r['year'] ])
    

def countLocations(entries:Entries)->int:
    loc = set()
    for r in entries:
        loc.add( r['location'] )
    return len(loc)
        
def invasion(entries: Entries, year:int)->Set[str]:
    before = set()
    after = set()
    for r in entries:
        sp =  r['species']
        if int(r['year']) <= year:
            before.add( sp )
        else:
            after.add(sp)
    return after-before

def locationyPerSpecies(entries:Entries)->Dict[str,Set[str]]:
    locSp: Dict[str,Set[str]] = {}
    for r in entries:
        loc = r['location']
        s =  r['species']
        if s not in locSp[s]:
            locSp[s] = set(loc)
        else:
            locSp[s].add( loc )
    return locSp
