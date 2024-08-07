# fast(ish) decode from Donald Adjeroh, Timothy Bell, Amar Mukherjee book

def bwt_encode(data):
    rotations = sorted(data[i:]+data[:i] for i in range(len(data)))
    return list(list(zip(*rotations))[-1]), rotations.index(data)

def bwt_decode(L, a):
    n = len(L)
    K = [0]*256
    M = [0]*256
    C = [0]*n
    for idx, num in enumerate(L):
        C[idx] = K[num]
        K[num] += 1
    total = 0
    for c in range(256):
        M[c] = total
        total += K[c]
    i = a
    Q = [0]*n
    for j in reversed(range(n)):
        Q[j] = L[i]
        i = C[i] + M[L[i]]
    return Q

def mtf_encode(data):
    byte_values = list(range(256))
    transformed = []
    for num in data:
        list_index = byte_values.index(num)
        transformed.append(list_index)
        byte_values.insert(0, byte_values.pop(list_index))
    return transformed

def mtf_decode(transformed):
    byte_values = list(range(256))
    data = []
    for num in transformed:
        data.append(byte_values[num])
        byte_values.insert(0, byte_values.pop(num))
    return data

if __name__ == '__main__':
    L, a = bwt_encode([1,0,2,0,3,0,4,0,5,0])
    print(L)
    print(bwt_decode(L, a))

    print(mtf_encode([1,0,1,1,1,1,4,3]))
    print(mtf_decode(mtf_encode([1,2,1,1,1,1,4,3])))