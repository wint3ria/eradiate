"""attrs-based utility classes and functions"""

import enum
from numbers import Number
from textwrap import dedent, indent

import attr
import numpy as np
import pint
import pinttr

from .._units import PhysicalQuantity
from .._units import unit_registry as ureg


class MetadataKey(enum.Enum):
    """Attribute metadata keys.

    These Enum values should be used as metadata attribute keys.
    """
    DOC = enum.auto()  #: Documentation for this field (str)
    TYPE = enum.auto()  #: Documented type for this field (str)
    DEFAULT = enum.auto()  #: Documented default value for this field (str)

# ------------------------------------------------------------------------------
#                           Attribute docs extension
# ------------------------------------------------------------------------------

@attr.s
class _FieldDoc:
    """Internal convenience class to store field documentation information."""
    doc = attr.ib(default=None)
    type = attr.ib(default=None)
    default = attr.ib(default=None)


def _eradiate_formatter(cls_doc, field_docs):
    """Appends a section on attributes to a class docstring.
    This docstring formatter is appropriate for Eradiate's current docstring
    format.

    Parameter ``cls_doc`` (str):
        Class docstring to extend.

    Parameter ``field_docs`` (dict[str, _FieldDoc]):
        Attributes documentation content.

    Returns → str:
        Updated class docstring.
    """
    # Do nothing if field is not documented
    if not field_docs:
        return cls_doc

    docstrings = []

    # Create docstring entry for each documented field
    for field_name, field_doc in field_docs.items():
        type_doc = f": {field_doc.type}" if field_doc.type is not None else ""
        default_doc = f" = {field_doc.default}" if field_doc.default is not None else ""

        docstrings.append(
            f"``{field_name}``{type_doc}{default_doc}\n"
            f"{indent(field_doc.doc, '    ')}\n"
        )

    # Assemble entries
    if docstrings:
        if cls_doc is None:
            cls_doc = ""

        return "\n".join((
            dedent(cls_doc.lstrip("\n")).rstrip(),
            "",
            ".. rubric:: Constructor arguments / instance attributes",
            "",
            "\n".join(docstrings),
            "",
        ))
    else:
        return cls_doc


def parse_docs(cls):
    """Extract attribute documentation and store documentation in a dunder
    class member.

    .. admonition:: Notes

       * Meant to be used as a class decorator.
       * Must be applied **after** ``@attr.s``.
       * Fields must be documented using :func:`documented`.

    This decorator will examine each ``attrs`` attribute and check its metadata
    for documentation content. It will then update the class's docstring
    based on this content.

    .. seealso:: :func:`documented`

    Parameter ``cls`` (class):
        Class whose attributes should be processed.

    Returns → class:
        Updated class.
    """
    formatter = _eradiate_formatter

    docs = {}
    for field in cls.__attrs_attrs__:
        if MetadataKey.DOC in field.metadata:
            # Collect field docstring
            docs[field.name] = _FieldDoc(doc=field.metadata[MetadataKey.DOC])

            # Collect field type
            if MetadataKey.TYPE in field.metadata:
                docs[field.name].type = field.metadata[MetadataKey.TYPE]
            else:
                docs[field.name].type = str(field.type)

            # Collect default value
            if MetadataKey.DEFAULT in field.metadata:
                docs[field.name].default = field.metadata[MetadataKey.DEFAULT]

    # Update docstring
    cls.__doc__ = formatter(cls.__doc__, docs)

    return cls


def get_doc(cls, attrib, field):
    """Fetch attribute documentation field. Requires fields metadata to be
    processed with :func:`documented`.

    Parameter ``cls`` (class):
        Class from which to get the attribute.

    Parameter ``attrib`` (str):
        Attribute from which to get the doc field.

    Parameter ``field`` ("doc" or "type" or "default"):
        Documentation field to query.

    Returns:
        Queried documentation content.

    Raises → ValueError:
        If the requested ``field`` is missing from the target attribute's
        metadata.

    Raises → ValueError:
        If the requested ``field`` is unsupported.
    """
    try:
        if field == "doc":
            return attr.fields_dict(cls)[attrib].metadata[MetadataKey.DOC]

        if field == "type":
            return attr.fields_dict(cls)[attrib].metadata[MetadataKey.TYPE]

        if field == "default":
            return attr.fields_dict(cls)[attrib].metadata[MetadataKey.DEFAULT]
    except KeyError:
        raise ValueError(f"{cls.__name__}.{attrib} has no documented field "
                         f"'{field}'")

    raise ValueError(f"unsupported attribute doc field {field}")


def documented(attrib, doc=None, type=None, default=None):
    """Declare an attrs field as documented.

    .. seealso:: :func:`parse_docs`

    Parameter ``doc`` (str or None):
        Docstring for the considered field. If set to ``None``, this function
        does nothing.

    Parameter ``type`` (str or None):
        Documented type for the considered field.

    Parameter ``default`` (str or None):
        Documented default value for the considered field.

    Returns → ``attrs`` attribute:
        ``attrib``, with metadata updated with documentation contents.
    """
    if doc is not None:
        attrib.metadata[MetadataKey.DOC] = doc

    if type is not None:
        attrib.metadata[MetadataKey.TYPE] = type

    if default is not None:
        attrib.metadata[MetadataKey.DEFAULT] = default

    return attrib


# ------------------------------------------------------------------------------
#                                 Converters
# ------------------------------------------------------------------------------

def converter_quantity(wrapped_converter):
    """Applies a converter to the magnitude of a :class:`pint.Quantity`."""

    def f(value):
        if isinstance(value, pint.Quantity):
            return wrapped_converter(value.magnitude) * value.units
        else:
            return wrapped_converter(value)

    return f


def converter_to_units(units):
    """Returns a converter which ensures that its passed value has units
    ``units``.

    .. seealso::

       :func:`ensure_units`
    """
    return lambda x: pinttr.converters.ensure_units(x, units)


def converter_or_auto(wrapped_converter):
    """Returns a converter which executes the wrapped converter if the converted
    value is not equal to ``"auto"``; otherwise returns ``"auto"``.
    """

    def f(value):
        if value == "auto":
            return value

        return wrapped_converter(value)

    return f

# ------------------------------------------------------------------------------
#                                 Validators
# ------------------------------------------------------------------------------

def validator_is_number(_, attribute, value):
    """Validates if ``value`` is a number.
    Raises a ``TypeError`` in case of failure.
    """
    if not isinstance(value, Number):
        raise TypeError(f"{attribute.name} must be a real number, "
                        f"got {value} which is a {value.__class__}")


def validator_is_vector3(instance, attribute, value):
    """Validates if ``value`` is convertible to a 3-vector."""
    return attr.validators.deep_iterable(
        member_validator=validator_is_number,
        iterable_validator=validator_has_len(3)
    )(instance, attribute, value)


def validator_is_string(_, attribute, value):
    """Validates if ``value`` is a ``str``.
    Raises a ``TypeError`` in case of failure.
    """
    if not isinstance(value, str):
        raise TypeError(f"{attribute} must be a string, "
                        f"got {value} which is a {type(value)}")


def validator_is_positive(_, attribute, value):
    """Validates if ``value`` is a positive number.
    Raises a ``ValueError`` in case of failure.
    """
    if value < 0.:
        raise ValueError(f"{attribute} must be positive or zero, got {value}")


def validator_all_positive(_, attribute, value):
    """Validates if all elements in ``value`` are positive number.
    Raises a ``ValueError`` in case of failure.
    """
    if isinstance(value, ureg.Quantity):
        value = value.magnitude
    if not np.all(np.array(value) >= 0):
        raise ValueError(f"{attribute} must be all positive or zero, got {value}")


def validator_path_exists(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing target. Raises a ``FileNotFoundError`` otherwise.
    """
    if not value.exists():
        raise FileNotFoundError(f"{attribute} points to '{str(value)}' "
                                f"(path does not exist)")


def validator_is_file(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing file. Raises a ``FileNotFoundError`` otherwise.
    """
    if not value.is_file():
        raise FileNotFoundError(f"{attribute} points to '{str(value)}' "
                                f"(not a file)")


def validator_is_dir(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing file. Raises a ``FileNotFoundError`` otherwise.
    """
    if not value.is_dir():
        raise FileNotFoundError(f"{attribute} points to '{str(value)}'"
                                f"(not a directory)")


def validator_has_len(size):
    """Generates a validator which validates if ``value`` is of length ``size``.
    The generated validator will raise a ``ValueError`` in
    case of failure.

    Parameter ``size`` (int):
        Size required to pass validation.

    Returns → callable(instance, attribute, value):
        Generated validator.
    """

    def f(_, attribute, value):
        if len(value) != size:
            raise ValueError(f"{attribute} must be have length {size}, "
                             f"got {value} of length {len(value)}")

    return f


def validator_has_quantity(quantity):
    """Validates if the validated value has a quantity field matching the
    ``quantity`` parameter."""

    quantity = PhysicalQuantity(quantity)

    def f(_, attribute, value):
        if value.quantity != quantity:
            raise ValueError(f"incompatible quantity '{value.quantity}' "
                             f"used to set field '{attribute.name}' "
                             f"(allowed: '{quantity}')")

    return f


def validator_quantity(wrapped_validator):
    """Applies a validator to either a value or its magnitude if it is a
    :class:`pint.Quantity` object.

    Parameter ``wrapped_validator`` (callable(instance, attribute, value)):
        A validator to wrap.

    Returns → callable(instance, attribute, value):
        Wrapped validator.
    """

    def f(instance, attribute, value):
        if isinstance(value, ureg.Quantity):
            return wrapped_validator(instance, attribute, value.magnitude)
        else:
            return wrapped_validator(instance, attribute, value)

    return f


def validator_or_auto(*wrapped_validators):
    """Validates if the validated value is ``"auto"`` or if all wrapped
    validators validate.

    .. note::
       ``wrapped_validators`` is variadic and can therefore be an arbitrary
       number of validators.
    """

    def f(instance, attribute, value):
        if value == "auto":
            return

        for validator in wrapped_validators:
            validator(instance, attribute, value)

    return f
