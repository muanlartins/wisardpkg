# BinInput

A compact bitpacked binary vector (8 bits per byte). The input type every WiSARD
model consumes. Models also accept plain `list[int]` and convert internally.

```python
import wisardpkg as wp

b = wp.BinInput(9)                       # 9 zeros
b = wp.BinInput([1,0,1,0,1,0,1,0,1])     # from a 0/1 list
b = wp.BinInput(other.data())            # from a serialized string

bit = b[3]                               # read
b[3] = 1                                 # write
len(b)                                   # 9
b.list()                                 # [1,0,1,0,1,0,1,0,1]
encoded = b.data()                       # Base64 string
b.extend([1,1,0])                        # append bits (also takes a BinInput)
```

Out-of-range `get`/`set` raise `Exception`.

For internal storage, the `data()` Base64 format, and source `file:line`, see
[data-structures](../../claude-docs/data-structures.md#bininput).
