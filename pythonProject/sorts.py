import time
from random import randint

"""
quick_sort_simple: 2.341970754000158 Space: #5
quick_sort: 2.9085828580000452  Space: #3
merge_sort: 4.653204685999754 Space: #4
heap_sort: 6.302412788000311 Space: #1
merge_sort_inplace: 12.273294753999835 Space: #2
"""

def test_sort(func):
    try:
        start = time.perf_counter()
        for _ in range(1000):
            x = [randint(-100, 100) for _ in range(1000)]
            y = func(x)
            if y is not None:
                x = y
            assert x == sorted(x), f"{func.__name__} failed to sort."
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start}")
    except AssertionError as e:
        print(f"{str(e)}")


def heap_sort(arr):
    """
    Sort a list in ascending order, using heap sort.

    time = O(n*lg_n)
    space = O(1)
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


def merge_sort(arr):
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
        # if len(arr) <= 16:
        #    _insertion_sort(arr)
        #    return

        mid = len(arr) // 2

        left = arr[:mid]
        right = arr[mid:]

        _merge_sort(left)
        _merge_sort(right)

        _merge(arr, left, right)

    _merge_sort(arr)


def quick_sort(arr):
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

    pivot = arr[randint(0, n-1)]

    lesser = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    return quick_sort_simple(lesser) + equal + quick_sort_simple(greater)



if __name__ == '__main__':
    test_sort(quick_sort)
    test_sort(quick_sort_simple)
    test_sort(merge_sort)
    test_sort(merge_sort_inplace)
    test_sort(heap_sort)
