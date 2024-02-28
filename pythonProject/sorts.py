import time
from random import randint

"""
quick_sort_in_place:
    Time: 0.00261	Time Ratio: 1.0	    Space: O(lg_n)~O(n) / low ~ medium
merge_sort_temp_arrays:
    Time: 0.00363	Time Ratio: 1.39	Space: O(n*lg_n) / medium
quick_sort_simple:
    Time: 0.00459	Time Ratio: 1.758	Space: O(n*lg_n)~(n^2) / medium ~ high
heap_sort_iterative:
    Time: 0.00539	Time Ratio: 2.062	Space: O(1) / very low
heap_sort_recursive:
    Time: 0.00552	Time Ratio: 2.113	Space: O(lg_n) / low
merge_sort_inplace:
    Time: 0.01181	Time Ratio: 4.521	Space: O(lg_n) / low
selection_sort:
    Time: 0.04606	Time Ratio: 17.635	Space: O(1) / very low
insertion_sort:
    Time: 0.05537	Time Ratio: 21.198	Space: O(1) / very low
bubble_sort:
    Time: 0.11289	Time Ratio: 43.218	Space: O(1) / very low
"""


def mean(arr):
    return sum(arr) / len(arr)


def test_sort(func):
    try:
        times = []

        for _ in range(100):
            x = [randint(-10000, 10000) for _ in range(1000)]

            start = time.perf_counter()
            y = func(x)
            end = time.perf_counter()
            times.append(end - start)

            if y is not None:
                x = y
            assert x == sorted(x), f"{func.__name__} failed to sort."

        return mean(times)

    except AssertionError as e:
        print(f"{str(e)}")
        return float("inf")

def heap_sort_iterative(arr):
    """
    Sort a list in ascending order, using heap sort.

    time = O(n*lg_n)
    space = O(1)
    """
    n = len(arr)
    if n <= 1:
        return

    def _heapify(arr, n, i):
        while i < n:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i

            if left < n and arr[left] > arr[largest]:
                largest = left
            if right < n and arr[right] > arr[largest]:
                largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                i = largest
            else:
                return

    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)

    for i in range(n - 1, 0 , -1):
        arr[i], arr[0] = arr[0], arr[i]
        _heapify(arr, i, 0)


def heap_sort_recursive(arr):
    """
    Sort a list in ascending order, using heap sort.

    time = O(n*lg_n)
    space = O(lg_n)
    """

    def _heapify(arr, n, i):
        left = 2 * i + 1
        right = 2 * i + 2

        greatest = i

        if left < n and arr[left] > arr[greatest]:
            greatest = left

        if right < n and arr[right] > arr[greatest]:
            greatest = right

        if greatest != i:
            arr[i], arr[greatest] = arr[greatest], arr[i]
            _heapify(arr, n, greatest)

    def _heap_sort(arr):
        n = len(arr)
        if n <= 1:
            return

        for i in range(n // 2 - 1, -1, -1):
            _heapify(arr, n, i)

        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            _heapify(arr, i, 0)

    _heap_sort(arr)


def merge_sort_inplace(arr):
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
        return

    def _insertion_sort(arr, start, end):
        if start >= end:
            return
        for i in range(start + 1, end + 1):
            key = arr[i]
            j = i - 1
            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def _get_next_gap(gap):
        if gap <= 1:
            return 0

        return (gap // 2) + (gap % 2)

    def _merge(arr, start, end):
        if start >= end:
            return

        gap = end - start + 1
        gap = _get_next_gap(gap)

        while gap > 0:
            i = start
            while gap + i <= end:
                j = gap + i
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
                i += 1
            gap = _get_next_gap(gap)

    def _merge_sort(arr, start, end):
        if start >= end:
            return
        if end - start <= 16:
            _insertion_sort(arr, start, end)
            return

        mid = start + (end - start) // 2

        _merge_sort(arr, start, mid)
        _merge_sort(arr, mid + 1, end)

        _merge(arr, start, end)

    _merge_sort(arr, 0, n - 1)


def merge_sort_temp_arrays(arr):
    """
    Sort a list in ascending order, using merge sort.

    time = O(n*lg_n)
    space = O(n*lg_n)
    """
    if len(arr) <= 1:
        return

    def _insertion_sort(arr):
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

    def _merge(arr, left, right):
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

    def _merge_sort(arr):
        if len(arr) <= 1:
            return
        if len(arr) <= 16:
            _insertion_sort(arr)
            return

        mid = len(arr) // 2

        left = arr[:mid]
        right = arr[mid:]

        _merge_sort(left)
        _merge_sort(right)

        _merge(arr, left, right)

    _merge_sort(arr)


def quick_sort_in_place(arr):
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
        return

    def _insertion_sort(arr, start, end):
        if start >= end:
            return

        for i in range(start + 1, end + 1):
            key = arr[i]
            j = i - 1

            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1

            arr[j + 1] = key

    def _get_median_three(arr, start, end):
        if end - start <= 3:
            return start

        mid = start + (end - start) // 2

        if arr[start] > arr[mid]:
            arr[start], arr[mid] = arr[mid], arr[start]
        if arr[mid] > arr[end]:
            arr[end], arr[mid] = arr[mid], arr[end]
        if arr[start] > arr[mid]:
            arr[start], arr[mid] = arr[mid], arr[start]

        return mid

    def _partition(arr, start, end):
        pivot_idx = _get_median_three(arr, start, end)

        arr[pivot_idx], arr[end] = arr[end], arr[pivot_idx]

        i = start
        for j in range(start, end):
            if arr[j] < arr[end]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[end] = arr[end], arr[i]

        return i

    def _quick_sort(arr, start, end):
        if start >= end:
            return
        if end - start <= 16:
            _insertion_sort(arr, start, end)
            return

        pivot_idx = _partition(arr, start, end)

        _quick_sort(arr, start, pivot_idx - 1)
        _quick_sort(arr, pivot_idx + 1, end)

    _quick_sort(arr, 0, n - 1)


def quick_sort_simple(arr):
    """
    Sorts a list in ascending order, using a variation of quick sort.

    *Not* in place.

    time = O(n^2) ( avg: O(n*lg_n) )
    space = O(n^2) ( avg: O(n*lg_n) )
    """
    n = len(arr)
    if n <= 1:
        return arr

    pivot = arr[randint(0, n - 1)]

    lesser = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    return quick_sort_simple(lesser) + equal + quick_sort_simple(greater)


def bubble_sort(arr):
    """
    Sort a list in ascending order, using bubble sort.

    time = O(n^2)
    space = O(1)
    """
    n = len(arr)

    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j + 1] < arr[j]:
                arr[j + 1], arr[j] = arr[j], arr[j + 1]
                swapped = True
        if not swapped:
            break


def selection_sort(arr):
    """
    Sort a list in ascending order, using selection sort.

    time = O(n^2)
    space = O(1)
    """
    n = len(arr)

    for i in range(n - 1):
        least = i
        for j in range(i + 1, n):
            if arr[j] < arr[least]:
                least = j
        arr[i], arr[least] = arr[least], arr[i]


def insertion_sort(arr):
    """
    Sort a list in ascending order, using insertion sort.

    time = O(n^2)
    space = O(1)
    """
    n = len(arr)

    for i in range(n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


if __name__ == '__main__':
    funcs = [
        (quick_sort_in_place, "O(lg_n)~O(n) / low ~ medium"),
        (quick_sort_simple, "O(n*lg_n)~(n^2) / medium ~ high"),
        (merge_sort_temp_arrays, "O(n*lg_n) / medium"),
        (merge_sort_inplace, "O(lg_n) / low"),
        (heap_sort_recursive, "O(lg_n) / low"),
        (heap_sort_iterative, "O(1) / very low"),
        (bubble_sort, "O(1) / very low"),
        (selection_sort, "O(1) / very low"),
        (insertion_sort, "O(1) / very low")
    ]

    times = []

    for f, space in funcs:
        times.append((f.__name__, test_sort(f), space))

    times.sort(key=lambda x: x[1])

    for f, time, space in times:
        time_ratio = time / times[0][1]
        print(f"{f}:\n\tTime: {round(time, 5)}\tTime Ratio: "
              f"{round(time_ratio, 3)}\tSpace: {space}")
