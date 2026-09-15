"""
Reading a name back from the string it prints as.
"""

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

# %% the inverse of printing a name


def test_a_name_survives_the_round_trip_through_its_string():
    """
    Both a prefixed and a plain name are read back from what they print as.
    """
    prefixed = PrefixedName(name="base_link", prefix="stretch_description")
    plain = PrefixedName(name="base_link")

    assert PrefixedName.from_string(str(prefixed)) == prefixed
    assert PrefixedName.from_string(str(plain)) == plain


def test_a_name_is_split_at_its_last_separator():
    """
    Everything before the last separator is the prefix, so a prefix holding one of its
    own stays whole.
    """
    assert PrefixedName.from_string("apartment/kitchen/sink") == PrefixedName(
        name="sink", prefix="apartment/kitchen"
    )
