LANES=['North','South','East','West']
CYCLE_TIME=90
MIN_GREEN=10

def compute_green_times(counts):
    normalized={lane:max(int(counts.get(lane,0)),0) for lane in LANES}
    total=sum(normalized.values())
    if total==0:
        return {lane:int(CYCLE_TIME/len(LANES)) for lane in LANES}
    remaining=CYCLE_TIME - MIN_GREEN*len(LANES)
    raw={}
    for lane in LANES:
        raw[lane]=MIN_GREEN + remaining*(normalized[lane]/total)
    rounded={lane:int(raw[lane]) for lane in LANES}
    delta=CYCLE_TIME - sum(rounded.values())
    for lane in sorted(LANES, key=lambda item: raw[item]-rounded[item], reverse=True):
        if delta==0:
            break
        rounded[lane]+=1
        delta-=1
    return rounded

if __name__=='__main__':
    counts={'North':10,'West':7,'South':5,'East':3}
    plan=compute_green_times(counts)
    print('counts=',counts)
    print('plan =',plan)
    print('sum=',sum(plan.values()))
