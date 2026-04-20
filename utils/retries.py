# retries.py

import logging
import time
import sys
from typing import Tuple, Type, Union, Iterable, Optional

# Set up logger for this library.
# Users can configure handlers and levels externally.
log = logging.getLogger(__name__)
# Add NullHandler to prevent "No handler found" warnings if user hasn't configured logging.
log.addHandler(logging.NullHandler())


class AttemptContext:
    """
    Context manager for a single retry attempt block.
    Used internally by RetryLoop.
    """

    def __init__(self, parent_loop: 'RetryLoop'):
        self._parent_loop = parent_loop
        # Expose attempt number for potential use within the block
        self.attempt_number = parent_loop._attempt_number
        self.max_retries = parent_loop._max_retries

    def __enter__(self):
        # Called when entering the 'with' block for an attempt.
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[object]) -> Optional[bool]:
        """Handles the outcome of the code block within the 'with' statement."""
        parent = self._parent_loop

        if exc_type is None:
            # Block completed without exception - SUCCESS!
            parent._register_success()
            # Return False (or None) - do not suppress anything, just signal clean exit
            return False

        # Check if the raised exception is one we should retry on
        if isinstance(exc_value, parent._exceptions_to_catch):
            parent._register_failure(exc_value)
            if parent._log_warnings:  # Check flag before logging
                log.warning(
                    f"Attempt {self.attempt_number}/{self.max_retries} failed with "
                    f"{exc_type.__name__}: {exc_value}. Retrying...")
            # Suppress the exception *within the context* so the for loop can continue
            return True  # VERY IMPORTANT: Suppress exception to allow next iteration
        else:
            # An unexpected exception occurred, do not suppress, let it propagate
            if parent._log_errors:  # Check flag before logging
                log.error(
                    f"Attempt {self.attempt_number}/{self.max_retries} failed with "
                    f"an unhandled exception type: {exc_type.__name__}: {exc_value}",
                    # exc_info=True # Optional: include traceback
                )
            # Return False (or None) to let the unhandled exception propagate out
            return False


class RetryLoop:
    """
    Iterator providing a retry loop with context management for each attempt.

    Usage:
        retry_loop = RetryLoop(...)
        try:
            for attempt_context in retry_loop: # loop yields context manager
                with attempt_context: # use context manager for the block
                    # Code block to retry:
                    print(f"Attempt {attempt_context.attempt_number}...")
                    # ... your code ...
            # If loop completes without raising, it succeeded.
        except Exception as e:
            # Handles final failure or unhandled exceptions.
            pass # Your handling
    """

    def __init__(self,
                 max_retries: int = 3,
                 delay: float = 0,
                 backoff: float = 1,
                 exceptions_to_catch: Union[Type[BaseException],
                                            Tuple[Type[BaseException],
                                                  ...]] = (Exception, ),
                 log_info: bool = False,
                 log_warnings: bool = False,
                 log_errors: bool = False):
        """
        Initializes the RetryLoop iterator.

        Args:
            max_retries: Maximum number of attempts (must be >= 1).
            delay: Initial delay in seconds between attempts (non-negative).
            backoff: Multiplier >= 1 for the delay after each failed attempt.
            exceptions_to_catch: Exception type(s) that should trigger a retry.
            log_info: If True, log informational messages (attempt start, waits).
            log_warnings: If True, log warnings on caught exceptions during retries.
            log_errors: If True, log an error if all retries are exhausted.
        """
        # Parameter Validation (same as previous version)
        if not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError("max_retries must be a positive integer")
        if not isinstance(delay, (int, float)) or delay < 0:
            raise ValueError("delay must be a non-negative number")
        if not isinstance(backoff, (int, float)) or backoff < 1:
            raise ValueError("backoff must be a number >= 1")
        if isinstance(exceptions_to_catch, type) and issubclass(
                exceptions_to_catch, BaseException):
            exceptions_to_catch = (exceptions_to_catch, )
        elif not (isinstance(exceptions_to_catch, tuple) and \
                  all(isinstance(e, type) and issubclass(e, BaseException) for e in exceptions_to_catch)):
            try:
                exceptions_to_catch_tuple = tuple(exceptions_to_catch)
                if not all(
                        isinstance(e, type) and issubclass(e, BaseException)
                        for e in exceptions_to_catch_tuple):
                    raise TypeError()
                exceptions_to_catch = exceptions_to_catch_tuple
            except TypeError:
                raise TypeError(
                    "exceptions_to_catch must be a type or tuple/iterable of types inheriting from BaseException"
                ) from None

        self._max_retries = max_retries
        self._delay = delay
        self._backoff = backoff
        self._exceptions_to_catch = exceptions_to_catch
        self._log_info = log_info
        self._log_warnings = log_warnings
        self._log_errors = log_errors

        # Internal state
        self._attempt_number: int = 0
        self._current_delay: float = 0.0
        self._last_exception: Optional[BaseException] = None
        self._succeeded: bool = False

    def __iter__(self) -> Iterable[AttemptContext]:
        """Returns the iterator object (self) and resets state."""
        self._attempt_number = 0
        self._current_delay = self._delay
        self._last_exception = None
        self._succeeded = False
        return self

    def __next__(self) -> AttemptContext:
        """Yields the context manager for the next attempt or handles loop end."""
        # Check if success was registered by the previous attempt's __exit__
        if self._succeeded:
            raise StopIteration  # Stop immediately if the last attempt succeeded

        # Check if retries are exhausted *before* starting the next attempt
        if self._attempt_number >= self._max_retries:
            # If we got here, the last attempt must have failed (otherwise _succeeded would be true)
            if self._last_exception:
                if self._log_errors:
                    log.error(
                        f"Operation failed after {self._max_retries} attempts. Raising last exception: "
                        f"{type(self._last_exception).__name__}: {self._last_exception}",
                        # exc_info=self._last_exception # Optional: include traceback
                    )
                # Raise the actual exception from the final failed attempt
                raise self._last_exception from self._last_exception
            else:
                # Should not happen if max_retries >= 1 and loop logic is correct
                raise StopIteration  # Or perhaps a different internal error

        # Apply delay if this is not the first attempt
        if self._attempt_number > 0:
            if self._current_delay > 0:
                if self._log_info:
                    log.info(
                        f"Waiting {self._current_delay:.2f}s before attempt {self._attempt_number + 1}..."
                    )
                time.sleep(self._current_delay)
                # Apply backoff for the *next* potential delay
                self._current_delay *= self._backoff

        # Increment attempt number for the upcoming attempt
        self._attempt_number += 1
        if self._log_info:
            log.info(
                f"--- Starting Retry Attempt {self._attempt_number}/{self._max_retries} ---"
            )

        # Yield a new context manager instance for this attempt
        return AttemptContext(self)

    # Internal methods used by AttemptContext via the parent_loop reference
    def _register_success(self):
        """Internal: Called by AttemptContext on successful block execution."""
        self._succeeded = True
        if self._log_info:
            log.info(f"--- Attempt {self._attempt_number} Succeeded ---")
        # No need to force stop, __next__ checks _succeeded on the next call

    def _register_failure(self, exception: BaseException):
        """Internal: Called by AttemptContext on caught exception."""
        self._last_exception = exception
        # Logging/warning happens within AttemptContext based on its flags check

