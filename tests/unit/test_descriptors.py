"""
Unit tests for the descriptors module.
"""

import pytest

from bituslabs_ds.descriptors import ListProperty


class TestListProperty:
    """Test the ListProperty descriptor."""

    def test_basic_functionality(self):
        """Test basic ListProperty functionality."""

        class TestClass:
            items = ListProperty("items")

        obj = TestClass()

        # Initially should be empty list
        assert obj.items == []

        # Set to a list
        obj.items = [1, 2, 3]
        assert obj.items == [1, 2, 3]

        # Set to a string (should be wrapped in list)
        obj.items = "test"
        assert obj.items == ["test"]

        # Set to None (should become empty list)
        obj.items = None
        assert obj.items == []

    def test_immutable_property(self):
        """Test immutable ListProperty."""

        class TestClass:
            items = ListProperty("items", immutable=True)

        obj = TestClass()

        # First assignment should work
        obj.items = [1, 2, 3]
        assert obj.items == [1, 2, 3]

        # Second assignment should raise AttributeError
        with pytest.raises(AttributeError, match="items is immutable and already set"):
            obj.items = [4, 5, 6]

    def test_mutable_property(self):
        """Test mutable ListProperty (default behavior)."""

        class TestClass:
            items = ListProperty("items", immutable=False)

        obj = TestClass()

        # Multiple assignments should work
        obj.items = [1, 2, 3]
        assert obj.items == [1, 2, 3]

        obj.items = [4, 5, 6]
        assert obj.items == [4, 5, 6]

    def test_string_wrapping(self):
        """Test that strings are wrapped in lists."""

        class TestClass:
            items = ListProperty("items")

        obj = TestClass()

        obj.items = "single_string"
        assert obj.items == ["single_string"]

        obj.items = ""
        assert obj.items == [""]

    def test_none_handling(self):
        """Test that None becomes empty list."""

        class TestClass:
            items = ListProperty("items")

        obj = TestClass()

        obj.items = None
        assert obj.items == []

    def test_multiple_instances_independence(self):
        """Test that multiple instances have independent properties."""

        class TestClass:
            items = ListProperty("items")

        obj1 = TestClass()
        obj2 = TestClass()

        obj1.items = [1, 2, 3]
        obj2.items = [4, 5, 6]

        assert obj1.items == [1, 2, 3]
        assert obj2.items == [4, 5, 6]
        assert obj1.items != obj2.items

    def test_list_modification(self):
        """Test that lists can be modified in place."""

        class TestClass:
            items = ListProperty("items")

        obj = TestClass()
        obj.items = [1, 2, 3]

        # Modify the list in place
        obj.items.append(4)
        assert obj.items == [1, 2, 3, 4]

        obj.items.extend([5, 6])
        assert obj.items == [1, 2, 3, 4, 5, 6]

    def test_immutable_after_modification(self):
        """Test that immutable property can be modified in place but not reassigned."""

        class TestClass:
            items = ListProperty("items", immutable=True)

        obj = TestClass()
        obj.items = [1, 2, 3]

        # In-place modification should work
        obj.items.append(4)
        assert obj.items == [1, 2, 3, 4]

        # Reassignment should fail
        with pytest.raises(AttributeError):
            obj.items = [5, 6, 7]

    def test_descriptor_access_from_class(self):
        """Test accessing the descriptor from the class."""

        class TestClass:
            items = ListProperty("items")

        # Accessing from class should return the descriptor
        descriptor = TestClass.items
        assert isinstance(descriptor, ListProperty)
        assert descriptor.private_name == "_items"
        assert descriptor.immutable is False

    def test_private_name_generation(self):
        """Test that private names are generated correctly."""
        descriptor = ListProperty("test_name")
        assert descriptor.private_name == "_test_name"

        descriptor = ListProperty("another_name", immutable=True)
        assert descriptor.private_name == "_another_name"
        assert descriptor.immutable is True
