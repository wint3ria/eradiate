import attr
import pytest

from eradiate import unit_registry as ureg
from eradiate.util.attrs import validator_quantity


def test_validator_quantity():
    v = validator_quantity(attr.validators.instance_of(float))

    # This should succeed
    v(None, None, 1.)
    v(None, None, ureg.Quantity(1., "km"))

    # This should fail
    @attr.s
    class Attribute:  # Tiny class to pass an appropriate attribute argument
        name = attr.ib()

    attribute = Attribute(name="attribute")

    with pytest.raises(TypeError): v(None, attribute, "1.")
    with pytest.raises(TypeError): v(None, attribute, ureg.Quantity("1.", "km"))
