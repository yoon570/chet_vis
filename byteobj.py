class Byte:
    def __init__(self, bv, st1, st2, st3):
        self.byte_val = bv
        self.stage1 = st1
        self.stage2 = st2
        self.stage3 = st3

    def __str__(self):
        return f"[{self.byte_val}], {self.stage1}, {self.stage2}, {self.stage3}"
    
class PageInfo:
    def __init__(self, dnary, occur):
        self.dictionary = dnary
        self.occurrences = occur