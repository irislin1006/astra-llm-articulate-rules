"""Classification tasks for rule articulation experiments."""

from .base import ClassificationTask
from .lexical_tasks import (
    AllLowercaseTask,
    AllUppercaseTask,
    ContainsExclamationTask,
    StartsWithVowelTask,
    EndsWithVowelTask,
    NoRepeatedLettersTask
)
from .numerical_tasks import (
    ContainsNumberTask,
    EvenDigitSumTask,
    ContainsPrimeTask,
    EvenWordCountTask
)

# Registry of all tasks (10 tasks total: 6 lexical + 4 numerical)
ALL_TASKS = [
    # Lexical (6 tasks)
    AllLowercaseTask,
    AllUppercaseTask,
    ContainsExclamationTask,
    StartsWithVowelTask,
    EndsWithVowelTask,
    NoRepeatedLettersTask,

    # Numerical (4 tasks)
    ContainsNumberTask,
    EvenDigitSumTask,
    ContainsPrimeTask,
    EvenWordCountTask,
]


def get_all_tasks(seed: int = 42):
    """
    Get instances of all tasks.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of task instances
    """
    return [TaskClass(seed=seed) for TaskClass in ALL_TASKS]


def get_task_by_id(task_id: str, seed: int = 42):
    """
    Get a specific task by its ID.

    Args:
        task_id: Task identifier
        seed: Random seed

    Returns:
        Task instance or None if not found
    """
    for TaskClass in ALL_TASKS:
        task = TaskClass(seed=seed)
        if task.task_id == task_id:
            return task
    return None


__all__ = [
    'ClassificationTask',
    'ALL_TASKS',
    'get_all_tasks',
    'get_task_by_id',
    # Lexical
    'AllLowercaseTask',
    'AllUppercaseTask',
    'ContainsExclamationTask',
    'StartsWithVowelTask',
    'EndsWithVowelTask',
    'NoRepeatedLettersTask',
    # Numerical
    'ContainsNumberTask',
    'EvenDigitSumTask',
    'ContainsPrimeTask',
    'EvenWordCountTask',
]
