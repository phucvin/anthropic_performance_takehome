.FFFFFFFF
======================================================================
FAIL: test_kernel_speedup (__main__.SpeedTests.test_kernel_speedup)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/app/tests/submission_tests.py", line 86, in test_kernel_speedup
    assert cycles() < BASELINE
           ^^^^^^^^^^^^^^^^^^^
AssertionError

======================================================================
FAIL: test_kernel_updated_starting_point (__main__.SpeedTests.test_kernel_updated_starting_point)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/app/tests/submission_tests.py", line 90, in test_kernel_updated_starting_point
    assert cycles() < 18532
           ^^^^^^^^^^^^^^^^
AssertionError

======================================================================
FAIL: test_opus45_11hr (__main__.SpeedTests.test_opus45_11hr)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/app/tests/submission_tests.py", line 111, in test_opus45_11hr
    assert cycles() < 1487
           ^^^^^^^^^^^^^^^
AssertionError

======================================================================
FAIL: test_opus45_2hr (__main__.SpeedTests.test_opus45_2hr)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/app/tests/submission_tests.py", line 103, in test_opus45_2hr
    assert cycles() < 1579
           ^^^^^^^^^^^^^^^
AssertionError

======================================================================
FAIL: test_opus45_casual (__main__.SpeedTests.test_opus45_casual)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/app/tests/submission_tests.py", line 99, in test_opus45_casual
    assert cycles() < 1790
           ^^^^^^^^^^^^^^^
AssertionError

======================================================================
FAIL: test_opus45_improved_harness (__main__.SpeedTests.test_opus45_improved_harness)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/app/tests/submission_tests.py", line 115, in test_opus45_improved_harness
    assert cycles() < 1363
           ^^^^^^^^^^^^^^^
AssertionError

======================================================================
FAIL: test_opus4_many_hours (__main__.SpeedTests.test_opus4_many_hours)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/app/tests/submission_tests.py", line 94, in test_opus4_many_hours
    assert cycles() < 2164
           ^^^^^^^^^^^^^^^
AssertionError

======================================================================
FAIL: test_sonnet45_many_hours (__main__.SpeedTests.test_sonnet45_many_hours)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/app/tests/submission_tests.py", line 107, in test_sonnet45_many_hours
    assert cycles() < 1548
           ^^^^^^^^^^^^^^^
AssertionError

----------------------------------------------------------------------
Ran 9 tests in 13.477s

FAILED (failures=8)
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Speedup over baseline:  1.0
