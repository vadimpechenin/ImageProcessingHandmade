import math

from common import Qtrac
from common.timeIt import timeit


PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


@timeit("simple ProcessPool")
def main():
    mainCycle()
    summarize(1)

def mainCycle():
        for number in PRIMES:
            print('%d is prime: %s' % (number, is_prime(number)))

def summarize(concurrency):
    message = "using {} processes".format(concurrency)
    Qtrac.report(message)
    print()

if __name__ == '__main__':
    main()