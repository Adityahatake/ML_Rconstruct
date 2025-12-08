# filter = keep only True elements
nums = [1,2,3]
out = list(map(lambda x: x*2, nums))
evens = list(filter(lambda x: x%2==0, nums))
print(evens)
