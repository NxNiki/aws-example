class ListProperty:
    """
    A descriptor that stores a list and optionally enforces immutability.

    Args:
        name (str): The name of the property.
        immutable (bool): If True, the value can only be set once.

    Behavior:
        - Strings are wrapped into single-element lists.
        - None becomes an empty list.
        - If immutable=True, any further assignment raises AttributeError.
    """

    def __init__(self, name: str, immutable: bool = False):
        self.private_name = f"_{name}"
        self.immutable = immutable

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.private_name):
            setattr(obj, self.private_name, [])
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if self.immutable and hasattr(obj, self.private_name):
            raise AttributeError(f"{self.private_name[1:]} is immutable and already set.")

        if value is None:
            value = []
        elif isinstance(value, str):
            value = [value]

        setattr(obj, self.private_name, value)


if __name__ == "__main__":

    class MyClass:
        cols = ListProperty("cols")

    a = MyClass()
    b = MyClass()
    a.cols.append(1)
    print(b.cols)
    print(a.cols)
