import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal


def test_melt() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    for _idv, _vv in (("A", ("B", "C")), (cs.string(), cs.integer())):
        melted_eager = df.melt(id_vars="A", value_vars=["B", "C"])
        assert melted_eager["value"].to_list() == [1, 3, 5, 2, 4, 6]

        melted_lazy = df.lazy().melt(id_vars="A", value_vars=["B", "C"])
        assert melted_lazy.collect()["value"].to_list() == [1, 3, 5, 2, 4, 6]

    melted = df.melt(id_vars="A", value_vars="B")
    assert melted["value"].to_list() == [1, 3, 5]
    n = 3

    for melted in [df.melt(), df.lazy().melt().collect()]:
        assert melted["variable"].to_list() == ["A"] * n + ["B"] * n + ["C"] * n
        assert melted["value"].to_list() == [
            "a",
            "b",
            "c",
            "1",
            "3",
            "5",
            "2",
            "4",
            "6",
        ]

    for melted in [
        df.melt(value_name="foo", variable_name="bar"),
        df.lazy().melt(value_name="foo", variable_name="bar").collect(),
    ]:
        assert melted["bar"].to_list() == ["A"] * n + ["B"] * n + ["C"] * n
        assert melted["foo"].to_list() == [
            "a",
            "b",
            "c",
            "1",
            "3",
            "5",
            "2",
            "4",
            "6",
        ]


def test_melt_projection_pd_7747() -> None:
    df = pl.LazyFrame(
        {
            "number": [1, 2, 1, 2, 1],
            "age": [40, 30, 21, 33, 45],
            "weight": [100, 103, 95, 90, 110],
        }
    )
    result = (
        df.with_columns(pl.col("age").alias("wgt"))
        .melt(id_vars="number", value_vars="wgt")
        .select("number", "value")
        .collect()
    )
    expected = pl.DataFrame(
        {
            "number": [1, 2, 1, 2, 1],
            "value": [40, 30, 21, 33, 45],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("cast", "use_string_cache"),
    [(False, None), (True, False), (True, True)],
)
def test_melt_categorical_10775(cast, use_string_cache) -> None:
    """
    TODO: summary.

    TODO: reduce the DataFrame size to make the test more readable.
    TODO: add more context to this docstring.
    # Case 1: Melt without casting, years are still of type pl.String
    # Case 2: Melt while casting years to pl.Categorical,
              with a local categorical string cache FIXME: shorter line
    # Case 3: Melt while casting years to pl.Categorical,
              with a local categorical string cache FIXME: shorter line
    """
    """
    >>> df_to_pivot
    shape: (4, 4)
    ┌──────┬───────┬───────────────────┬───────────────────┐
    │ race ┆ sex   ┆ 2008              ┆ 2009              │
    │ ---  ┆ ---   ┆ ---               ┆ ---               │
    │ str  ┆ str   ┆ str               ┆ str               │
    ╞══════╪═══════╪═══════════════════╪═══════════════════╡
    │ road ┆ man   ┆ Alessandro Ballan ┆ Cadel Evans       │
    │ itt  ┆ man   ┆ Bert Grabsch      ┆ Fabian Cancellara │
    │ road ┆ woman ┆ Nicole Cooke      ┆ Tatiana Guderzo   │
    │ itt  ┆ woman ┆ Amber Neben       ┆ Kristin Armstrong │
    └──────┴───────┴───────────────────┴───────────────────┘
    """
    df_to_pivot = pl.from_records(
        [
            {
                "race": "road",
                "sex": "man",
                "2008": "Alessandro Ballan",
                "2009": "Cadel Evans",
            },
            {
                "race": "itt",
                "sex": "man",
                "2008": "Bert Grabsch",
                "2009": "Fabian Cancellara",
            },
            {
                "race": "road",
                "sex": "woman",
                "2008": "Nicole Cooke",
                "2009": "Tatiana Guderzo",
            },
            {
                "race": "itt",
                "sex": "woman",
                "2008": "Amber Neben",
                "2009": "Kristin Armstrong",
            },
        ]
    )

    """
    >>> expected
    shape: (8, 4)
    ┌───────┬──────┬──────┬───────────────────┐
    │ sex   ┆ race ┆ year ┆ winner            │
    │ ---   ┆ ---  ┆ ---  ┆ ---               │
    │ str   ┆ str  ┆ str  ┆ str               │
    ╞═══════╪══════╪══════╪═══════════════════╡
    │ man   ┆ road ┆ 2008 ┆ Alessandro Ballan │
    │ man   ┆ itt  ┆ 2008 ┆ Bert Grabsch      │
    │ woman ┆ road ┆ 2008 ┆ Nicole Cooke      │
    │ woman ┆ itt  ┆ 2008 ┆ Amber Neben       │
    │ man   ┆ road ┆ 2009 ┆ Cadel Evans       │
    │ man   ┆ itt  ┆ 2009 ┆ Fabian Cancellara │
    │ woman ┆ road ┆ 2009 ┆ Tatiana Guderzo   │
    │ woman ┆ itt  ┆ 2009 ┆ Kristin Armstrong │
    └───────┴──────┴──────┴───────────────────┘
    """
    expected = pl.DataFrame(
        {
            "sex": ["man", "man", "woman", "woman", "man", "man", "woman", "woman"],
            "race": ["road", "itt", "road", "itt", "road", "itt", "road", "itt"],
            "year": ["2008", "2008", "2008", "2008", "2009", "2009", "2009", "2009"],
            "winner": [
                "Alessandro Ballan",
                "Bert Grabsch",
                "Nicole Cooke",
                "Amber Neben",
                "Cadel Evans",
                "Fabian Cancellara",
                "Tatiana Guderzo",
                "Kristin Armstrong",
            ],
        }
    )

    if cast:
        df_to_pivot = df_to_pivot.with_columns(cs.matches("\\d+").cast(pl.Categorical))

    if use_string_cache is not None:
        pl.enable_string_cache(use_string_cache)

    df_melted = (
        df_to_pivot.melt(
            id_vars=["sex", "race"],
            variable_name="year",
            value_name="winner",
        )
        # Cast back to pl.String for comparison purposes.
        .cast(pl.String)
    )
    assert_frame_equal(expected, df_melted)
