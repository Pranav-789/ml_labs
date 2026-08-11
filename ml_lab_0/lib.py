def total(nums: list[int | float])-> int | float:
    ans = 0
    n = len(nums)
    
    for num in nums:
        ans = ans + num

    return ans

def squared_total(nums: list[int | float]) -> int | float:
    ans = 0

    for num in nums:
        ans = ans + (num*num)

    return ans

def mean(nums: list[int | float]) -> int | float:
    n = len(nums)

    return total(nums)/n

def mode(nums: list[int | float]) -> int | float:
    freq = dict()
    maxf = 0
    mode_value = None

    for num in nums:
        freq[num] = freq.get(num, 0) + 1
        if freq[num] > maxf:
            mode_value = num
            maxf = freq[num]

    return mode_value

def median(nums: list[int | float]) -> int | float:
    n = len(nums)
    nums = sorted(nums)
    if n % 2 == 0:
        return (nums[n//2] + nums[n//2 - 1])/2
    else:
        return nums[n//2]

def variance(nums: list[int | float]) -> int | float:
    return (squared_total(nums) - (mean(nums)**2)*len(nums))/(len(nums)-1)

def std(nums: list[int | float]) -> int | float:
    return variance(nums)**(0.5)

def range_list(nums: list[int | float]) -> int | float:
    return max(nums) - min(nums)

def mean_deviation(nums: list[int | float]) -> int | float:
    mean_value = mean(nums)

    total_dev = 0
    for num in nums:
        total_dev = total_dev + abs(mean_value-num)
    return total_dev / len(nums)

def coef_range(nums: list[int | float]) -> int | float:
    return (max(nums) - min(nums)) / (max(nums) + min(nums))

def coef_var(nums: list[int | float]) -> int | float:
    return (std(nums)/mean(nums))*100

def quartile(nums: list[int | float]) -> list[int | float]:
    n = len(nums)
    q2 = median(nums)

    if(n % 2 == 0):
        lower = nums[: n//2]
        upper = nums[n//2 : ]
    else:
        lower = nums[: n//2]
        upper = nums[n//2+1:]

    q1 = median(lower)
    q3 = median(upper)

    return [q1, q2, q3]