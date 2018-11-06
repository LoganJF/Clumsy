# Clumsy
Clumsy (originally cml_lib) is a collection of various scripts used in the computational memory lab's python analysis.

What is Clumsy?
-------------

more details about it here blah blah blah:

```python

from Clumsy.signal import rolling_window
from typing import Iterator

def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b
```
See the documentation for more examples.

For Python 2.7, the standard annotations are written as comments:
```python
def is_palindrome(s):
    # type: (str) -> bool
    return s == s[::-1]
```

Clumsy is in development; some features are missing and there are bugs.
See 'Development status' below.

Requirements
------------
pyedflib

mne

ptsa

scipy

numpy

xarray

pandas

six

cmlreaders

numba

visbrain


Quick start
-----------

Clumsy can be installed using pip:

    $ pip install Clumsy

If you want to run the latest version of the code, you can install from git:

    $mkdir Clumsy
    $cd Clumsy
    $git clone https://github.com/LoganJF/Clumsy.git .
    $python setup.py install


Development status
------------------

Issue tracker
-------------
