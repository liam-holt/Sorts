import time
from random import randint, choice
from copy import deepcopy

""" 
 n = 100000
Quick Sort w/ Insertion:
    0.34643 seconds on average.		time ratio: 1.0
    time: O(n*lg_n)~O(n^2)		space: O(lg_n)~O(n)
    flow: Recursive		stability: Unstable		movement: Inplace

Quick Sort w/ Insertion:
    0.3517 seconds on average.		time ratio: 1.015
    time: O(n*lg_n)~O(n^2)		space: O(lg_n)~O(n)
    flow: Iterative		stability: Unstable		movement: Inplace

Quick Sort (Simple):
    0.37288 seconds on average.		time ratio: 1.076
    time: O(n*lg_n)~O(n^2)		space: O(n*lg_n)~O(n^2)
    flow: Recursive		stability: Stable		movement: Mobile

Merge Sort w/ Insertion:
    0.50827 seconds on average.		time ratio: 1.467
    time: O(n*lg_n)		space: O(n*lg_n)
    flow: Recursive		stability: Stable		movement: Mobile

Heap Sort:
    0.96611 seconds on average.		time ratio: 2.789
    time: O(n*lg_n)		space: O(1)
    flow: Iterative		stability: Unstable		movement: Inplace

Heap Sort:
    1.00205 seconds on average.		time ratio: 2.892
    time: O(n*lg_n)		space: O(lg_n)
    flow: Recursive		stability: Unstable		movement: Inplace

Merge Sort w/ Insertion:
    3.65459 seconds on average.		time ratio: 10.549
    time: O(n*lg^2_n)		space: O(lg_n)
    flow: Recursive		stability: Stable		movement: Inplace
"""

"""
 n = 1000
Quick Sort w/ Insertion:
    0.00212 seconds on average.		time ratio: 1.0
    time: O(n*lg_n)~O(n^2)		space: O(lg_n)~O(n)
    flow: Iterative		stability: Unstable		movement: Inplace
    
Quick Sort w/ Insertion:
    0.00213 seconds on average.		time ratio: 1.0
    time: O(n*lg_n)~O(n^2)		space: O(lg_n)~O(n)
    flow: Recursive		stability: Unstable		movement: Inplace

Quick Sort (Simple):
    0.00244 seconds on average.		time ratio: 1.152
    time: O(n*lg_n)~O(n^2)		space: O(n*lg_n)~O(n^2)
    flow: Recursive		stability: Stable		movement: Mobile

Merge Sort w/ Insertion:
    0.00296 seconds on average.		time ratio: 1.395
    time: O(n*lg_n)		space: O(n*lg_n)
    flow: Recursive		stability: Stable		movement: Mobile

Heap Sort:
    0.0053 seconds on average.		time ratio: 2.5
    time: O(n*lg_n)		space: O(1)
    flow: Iterative		stability: Unstable		movement: Inplace

Heap Sort:
    0.00553 seconds on average.		time ratio: 2.612
    time: O(n*lg_n)		space: O(lg_n)
    flow: Recursive		stability: Unstable		movement: Inplace

Merge Sort w/ Insertion:
    0.01151 seconds on average.		time ratio: 5.436
    time: O(n*lg^2_n)		space: O(lg_n)
    flow: Recursive		stability: Stable		movement: Inplace

Selection Sort:
    0.0434 seconds on average.		time ratio: 20.49
    time: O(n^2)		space: O(1)
    flow: Iterative		stability: Unstable		movement: Inplace

Insertion Sort:
    0.05298 seconds on average.		time ratio: 25.012
    time: O(n^2)		space: O(1)
    flow: Iterative		stability: Stable		movement: Inplace

Bubble Sort:
    0.10777 seconds on average.		time ratio: 50.879
    time: O(n^2)		space: O(1)
    flow: Iterative		stability: Stable		movement: Inplace
"""

LARGE_DATA = [  # n = 1,000,000
    [
        randint(-1_000_000, 1_000_000)
        for _ in range(100_000)
    ] for _ in range(10)
]

SMALL_DATA = [  # n = 1,000
    [
        randint(-1_000_000, 1_000_000)
        for _ in range(1_000)
    ] for _ in range(100)
]


class Sorter:
    def __init__(
            self, name, time, space, is_inplace, is_stable,
            is_recursive, function
    ):
        self.name = name
        self.time = time
        self.space = space
        self.is_inplace = is_inplace
        self.is_stable = is_stable
        self.is_recursive = is_recursive
        self.timed = float("inf")

        self.sort = function


def mean(arr):
    print(len(arr))
    return sum(arr) / len(arr)


def time_sort(func, datasets):
    try:
        times = []

        for dataset in datasets:
            start = time.perf_counter()
            dataset = func(dataset)
            end = time.perf_counter()

            times.append(end - start)

            assert dataset == sorted(dataset), \
                f"{func.__name__} failed to sort."

        return mean(times)

    except AssertionError as e:
        print(f"{e}")
        return float("inf")


def median_of_three(arr, start, end):  # for pivot selection
    mid = start + (end - start) // 2

    if arr[start] > arr[mid]:
        arr[start], arr[mid] = arr[mid], arr[start]
    if arr[mid] > arr[end]:
        arr[end], arr[mid] = arr[mid], arr[end]
    if arr[start] > arr[mid]:
        arr[start], arr[mid] = arr[mid], arr[start]

    return mid


def partition(arr, start, end):
    pivot_idx = median_of_three(arr, start, end)
    arr[end], arr[pivot_idx] = arr[pivot_idx], arr[end]

    i = start
    for j in range(start, end):
        if arr[j] < arr[end]:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[end] = arr[end], arr[i]

    return i


def heapify_iterative(arr, n, i):
    while i < n:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            i = largest
        else:
            return


def heapify_recursive(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify_recursive(arr, n, largest)


def next_gap(gap):  # for shell swapping
    if gap <= 1:
        return 0

    return gap // 2 + gap % 2


def insertion_sort(arr, start, end):
    if start >= end:
        return

    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        while j >= start and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr


def insertion_sort_slice(arr):
    n = len(arr)
    if n <= 1:
        return

    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr


def merge_slices(arr, left, right):
    i = j = k = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1


def merge_inplace(arr, start, end):
    if start >= end:
        return

    gap = end - start + 1
    gap = next_gap(gap)

    while gap > 0:
        i = start
        while gap + i <= end:
            j = gap + i
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
            i += 1
        gap = next_gap(gap)


def print_sorts(sorters, datasets):
    for sorter in sorters:
        x = deepcopy(datasets)
        sorter.timed = time_sort(sorter.sort, x)

    sorters.sort(key=lambda sorter: sorter.timed)

    print(f" n = {len(datasets[0])}")

    for sorter in sorters:
        time_ratio = sorter.timed / sorters[0].timed
        is_recursive = "Recursive" if sorter.is_recursive else "Iterative"
        is_stable = "Stable" if sorter.is_stable else "Unstable"
        is_inplace = "Inplace" if sorter.is_inplace else "Mobile"
        print(
            f"{sorter.name}:\n"
            f"    {round(sorter.timed, 5)} seconds on average.\t\ttime ratio: "
            f"{round(time_ratio, 3)}\n"
            f"    time: {sorter.time}\t\tspace: {sorter.space}\n"
            f"    flow: {is_recursive}\t\tstability: "
            f"{is_stable}\t\tmovement: {is_inplace}\n"
        )


def heap_sort_iterative_inplace_unstable(arr):
    """
    Sort a list in ascending order, using heap sort.

    time = O(n*lg_n)
    space = O(1)
    """
    n = len(arr)
    if n <= 1:
        return arr

    for i in range(n // 2 - 1, -1, -1):
        heapify_iterative(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify_iterative(arr, i, 0)

    return arr


def heap_sort_recursive_inplace_unstable(arr):
    """
    Sort a list in ascending order, using heap sort.

    time = O(n*lg_n)
    space = O(lg_n)
    """

    n = len(arr)
    if n <= 1:
        return arr

    for i in range(n // 2 - 1, -1, -1):
        heapify_recursive(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify_recursive(arr, i, 0)

    return arr


def merge_sort_recursive_inplace_stable_insertion(arr):
    """
    Sorts a list in ascending order, using merge sort.

    In-place variation of merge sort.

    time = O(n*lg^2_n)
    space = O(lg_n)

    :param arr: The list to sort.
    :return: None
    """
    n = len(arr)
    if n <= 1:
        return arr

    def _merge_sort(arr, start, end):
        if start >= end:
            return arr
        if end - start <= 16:
            insertion_sort(arr, start, end)
            return arr

        mid = start + (end - start) // 2

        _merge_sort(arr, start, mid)
        _merge_sort(arr, mid + 1, end)

        merge_inplace(arr, start, end)

    _merge_sort(arr, 0, n - 1)
    return arr


def merge_sort_recursive_mobile_stable_insertion(arr):
    """
    Sort a list in ascending order, using merge sort.

    time = O(n*lg_n)
    space = O(n*lg_n)
    """
    n = len(arr)
    if n <= 1:
        return arr

    if n <= 16:
        insertion_sort_slice(arr)
        return arr

    mid = len(arr) // 2

    left = arr[:mid]
    right = arr[mid:]

    merge_sort_recursive_mobile_stable_insertion(left)
    merge_sort_recursive_mobile_stable_insertion(right)

    merge_slices(arr, left, right)

    return arr


def quicksort_iterative_inplace_unstable_insertion(arr):
    n = len(arr)
    if n <= 1:
        return arr

    stack = []

    stack.append((0, n - 1))

    while len(stack) > 0:
        start, end = stack.pop()

        if start >= end:
            continue
        if end - start <= 16:
            insertion_sort(arr, start, end)
            continue

        pivot_index = partition(arr, start, end)
        stack.append((start, pivot_index - 1))
        stack.append((pivot_index + 1, end))

    return arr


def quick_sort_recursive_inplace_unstable_insertion(arr):
    """
    Sort a list in ascending order, using quick sort.

    Use median-of-three pivoting. In-place sort.

    time = O(n^2) ( avg: O(n*lg_n) )
    space = O(n) ( avg:  O(lg_n) )

    :param arr: The list to sort.
    :return: None
    """
    n = len(arr)
    if n <= 1:
        return arr

    def _quick_sort(arr, start, end):
        if start >= end:
            return
        if end - start <= 16:
            insertion_sort(arr, start, end)
            return

        pivot_idx = partition(arr, start, end)

        _quick_sort(arr, start, pivot_idx - 1)
        _quick_sort(arr, pivot_idx + 1, end)

    _quick_sort(arr, 0, n - 1)
    return arr


def quick_sort_recursive_mobile_stable(arr):
    """
    Sorts a list in ascending order, using a variation of quick sort.

    *Not* in place.

    time = O(n^2) ( avg: O(n*lg_n) )
    space = O(n^2) ( avg: O(n*lg_n) )
    """
    n = len(arr)
    if n <= 1:
        return arr

    pivot = choice(arr)

    lesser = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    return quick_sort_recursive_mobile_stable(
        lesser
    ) + equal + quick_sort_recursive_mobile_stable(greater)


def bubble_sort_iterative_inplace_stable(arr):
    """
    Sort a list in ascending order, using bubble sort.

    time = O(n^2)
    space = O(1)
    """
    n = len(arr)
    if n <= 1:
        return arr

    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j + 1] < arr[j]:
                arr[j + 1], arr[j] = arr[j], arr[j + 1]
                swapped = True
        if not swapped:
            break

    return arr


def selection_sort_iterative_inplace_unstable(arr):
    """
    Sort a list in ascending order, using selection sort.

    time = O(n^2)
    space = O(1)
    """
    n = len(arr)
    if n <= 1:
        return arr

    for i in range(n - 1):
        least = i
        for j in range(i + 1, n):
            if arr[j] < arr[least]:
                least = j
        arr[i], arr[least] = arr[least], arr[i]

    return arr


def insertion_sort_iterative_inplace_stable(arr):
    """
    Sort a list in ascending order, using insertion sort.

    time = O(n^2)
    space = O(1)
    """
    n = len(arr)
    if n <= 1:
        return arr

    for i in range(n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr

# introsort
def intro_sort_recursive_inplace_unstable(arr):
    n = len(arr)
    max_depth = 2 * (n.bit_length() - 1)
    size_threshold = 16

    def insertion_sort(arr, start, end):
        for i in range(start + 1, end + 1):
            key = arr[i]
            j = i - 1
            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def partition(arr, start, end):
        pivot = arr[end]
        i = start
        for j in range(start, end):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[end] = arr[end], arr[i]
        return i

    def quick_sort(arr, start, end, depth):
        if end - start <= size_threshold:
            insertion_sort(arr, start, end)
        elif depth >= max_depth:
            heap_sort(arr, start, end)
        else:
            pivot = partition(arr, start, end)
            quick_sort(arr, start, pivot - 1, depth + 1)
            quick_sort(arr, pivot + 1, end, depth + 1)

    def heapify(arr, n, i, start):
        j = start + i
        m = start + n
        largest = j
        left = start + 2 * i + 1
        right = start + 2 * i + 2
        if left < m and arr[largest] < arr[left]:
            largest = left
        if right < m and arr[largest] < arr[right]:
            largest = right
        if largest != j:
            arr[j], arr[largest] = arr[largest], arr[j]
            heapify(arr, n, largest - start, start)

    def heap_sort(arr, start, end):
        n = end - start + 1
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i, start)
        for i in range(n - 1, 0, -1):
            arr[start + i], arr[start] = arr[start], arr[start + i]
            heapify(arr, i, 0)

    quick_sort(arr, 0, n - 1, 0)

# timsort

# mergesort

# quicksort

if __name__ == '__main__':
    sort_large_data = [
        {
            "name": "Quick Sort (Simple)", "time": "O(n*lg_n)~O(n^2)",
            "space": "O(n*lg_n)~O(n^2)", "is_inplace": False,
            "is_stable": True,
            "is_recursive": True,
            "function": quick_sort_recursive_mobile_stable
        },
        {
            "name": "Quick Sort w/ Insertion", "time": "O(n*lg_n)~O(n^2)",
            "space": "O(lg_n)~O(n)", "is_inplace": True, "is_stable": False,
            "is_recursive": True,
            "function": quick_sort_recursive_inplace_unstable_insertion
        },
        {
            "name": "Quick Sort w/ Insertion", "time": "O(n*lg_n)~O(n^2)",
            "space": "O(lg_n)~O(n)", "is_inplace": True, "is_stable": False,
            "is_recursive": False,
            "function": quicksort_iterative_inplace_unstable_insertion
        },
        {
            "name": "Merge Sort w/ Insertion", "time": "O(n*lg_n)",
            "space": "O(n*lg_n)", "is_inplace": False, "is_stable": True,
            "is_recursive": True,
            "function": merge_sort_recursive_mobile_stable_insertion
        },
        {
            "name": "Merge Sort w/ Insertion", "time": "O(n*lg^2_n)",
            "space": "O(lg_n)", "is_inplace": True, "is_stable": True,
            "is_recursive": True,
            "function": merge_sort_recursive_inplace_stable_insertion
        },
        {
            "name": "Heap Sort", "time": "O(n*lg_n)",
            "space": "O(lg_n)", "is_inplace": True, "is_stable": False,
            "is_recursive": True,
            "function": heap_sort_recursive_inplace_unstable
        },
        {
            "name": "Heap Sort", "time": "O(n*lg_n)",
            "space": "O(1)", "is_inplace": True, "is_stable": False,
            "is_recursive": False,
            "function": heap_sort_iterative_inplace_unstable
        }
    ]

    sort_small_data = [
        {
            "name": "Bubble Sort", "time": "O(n^2)",
            "space": "O(1)", "is_inplace": True, "is_stable": True,
            "is_recursive": False,
            "function": bubble_sort_iterative_inplace_stable
        },
        {
            "name": "Selection Sort", "time": "O(n^2)",
            "space": "O(1)", "is_inplace": True, "is_stable": False,
            "is_recursive": False,
            "function": selection_sort_iterative_inplace_unstable
        },
        {
            "name": "Insertion Sort", "time": "O(n^2)",
            "space": "O(1)", "is_inplace": True, "is_stable": True,
            "is_recursive": False,
            "function": insertion_sort_iterative_inplace_stable
        }
    ]

    large_sorters = [Sorter(**data) for data in sort_large_data]
    small_sorters = large_sorters + [Sorter(**data) for data in
        sort_small_data]

    # print_sorts(large_sorters, LARGE_DATA)
    # print_sorts(small_sorters, SMALL_DATA)

