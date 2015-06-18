import collections
import os

import six

from fuel import config


# See http://python3porting.com/differences.html#buffer
if six.PY3:
    buffer_ = memoryview
else:
    buffer_ = buffer  # noqa


def find_in_data_path(filename):
    """Searches for a file within Fuel's data path.

    This function loops over all paths defined in Fuel's data path and
    returns the first path in which the file is found.

    Parameters
    ----------
    filename : str
        Name of the file to find.

    Returns
    -------
    file_path : str
        Path to the first file matching `filename` found in Fuel's
        data path.

    Raises
    ------
    IOError
        If the file doesn't appear in Fuel's data path.

    """
    for path in config.data_path.split(os.path.pathsep):
        path = os.path.expanduser(os.path.expandvars(path))
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            return file_path
    raise IOError("{} not found in Fuel's data path".format(filename))


def lazy_property_factory(lazy_property):
    """Create properties that perform lazy loading of attributes."""
    def lazy_property_getter(self):
        if not hasattr(self, '_' + lazy_property):
            self.load()
        if not hasattr(self, '_' + lazy_property):
            raise ValueError("{} wasn't loaded".format(lazy_property))
        return getattr(self, '_' + lazy_property)

    def lazy_property_setter(self, value):
        setattr(self, '_' + lazy_property, value)

    return lazy_property_getter, lazy_property_setter


def do_not_pickle_attributes(*lazy_properties):
    r"""Decorator to assign non-pickable properties.

    Used to assign properties which will not be pickled on some class.
    This decorator creates a series of properties whose values won't be
    serialized; instead, their values will be reloaded (e.g. from disk) by
    the :meth:`load` function after deserializing the object.

    The decorator can be used to avoid the serialization of bulky
    attributes. Another possible use is for attributes which cannot be
    pickled at all. In this case the user should construct the attribute
    himself in :meth:`load`.

    Parameters
    ----------
    \*lazy_properties : strings
        The names of the attributes that are lazy.

    Notes
    -----
    The pickling behavior of the dataset is only overridden if the
    dataset does not have a ``__getstate__`` method implemented.

    Examples
    --------
    In order to make sure that attributes are not serialized with the
    dataset, and are lazily reloaded after deserialization by the
    :meth:`load` in the wrapped class. Use the decorator with the names of
    the attributes as an argument.

    >>> from fuel.datasets import Dataset
    >>> @do_not_pickle_attributes('features', 'targets')
    ... class TestDataset(Dataset):
    ...     def load(self):
    ...         self.features = range(10 ** 6)
    ...         self.targets = range(10 ** 6)[::-1]

    """
    def wrap_class(cls):
        if not hasattr(cls, 'load'):
            raise ValueError("no load method implemented")

        # Attach the lazy loading properties to the class
        for lazy_property in lazy_properties:
            setattr(cls, lazy_property,
                    property(*lazy_property_factory(lazy_property)))

        # Delete the values of lazy properties when serializing
        if not hasattr(cls, '__getstate__'):
            def __getstate__(self):
                serializable_state = self.__dict__.copy()
                for lazy_property in lazy_properties:
                    attr = serializable_state.get('_' + lazy_property)
                    # Iterators would lose their state
                    if isinstance(attr, collections.Iterator):
                        raise ValueError("Iterators can't be lazy loaded")
                    serializable_state.pop('_' + lazy_property, None)
                return serializable_state
            setattr(cls, '__getstate__', __getstate__)

        return cls
    return wrap_class
