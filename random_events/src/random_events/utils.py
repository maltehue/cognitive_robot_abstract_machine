from abc import abstractmethod
from typing_extensions import Dict, Any, Self


class CPPWrapper:
    """
    Class for wrapping C++ objects.
    """

    _cpp_object: Any
    """
    The C++ object that this class wraps.
    """

    @abstractmethod
    def _from_cpp(self, cpp_object: Any) -> Self:
        """
        Create a new instance of this class from a C++ object.

        This method should also add fields that are python only to the instance that is created.
        This cannot be a class method since the values of the python-only fields are instance-specific.
        """
        raise NotImplementedError
