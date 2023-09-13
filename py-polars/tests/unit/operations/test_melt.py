import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal


def test_melt() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    for _idv, _vv in (("A", ("B", "C")), (cs.string(), cs.integer())):
        melted_eager = df.melt(id_vars="A", value_vars=["B", "C"])
        assert all(melted_eager["value"] == [1, 3, 5, 2, 4, 6])

        melted_lazy = df.lazy().melt(id_vars="A", value_vars=["B", "C"])
        assert all(melted_lazy.collect()["value"] == [1, 3, 5, 2, 4, 6])

    melted = df.melt(id_vars="A", value_vars="B")
    assert all(melted["value"] == [1, 3, 5])
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


# def test_melt_categorical() -> None:
#     """https://github.com/pola-rs/polars/issues/10775"""
#
#     # Build the dataframe to melt
#     df = pl.from_records(
#         [
#             {"race": "road", "sex": "man", "2008": "Alessandro Ballan", "2009": "Cadel Evans"},
#             {"race": "itt", "sex": "man", "2008": "Bert Grabsch", "2009": "Fabian Cancellara"},
#             {"race": "road", "sex": "woman", "2008": "Nicole Cooke", "2009": "Tatiana Guderzo"},
#             {"race": "itt", "sex": "woman", "2008": "Amber Neben", "2009": "Kristin Armstrong"},
#         ]
#     )
#     >>> df
#     shape: (4, 4)
#     ┌──────┬───────┬───────────────────┬───────────────────┐
#     │ race ┆ sex   ┆ 2008              ┆ 2009              │
#     │ ---  ┆ ---   ┆ ---               ┆ ---               │
#     │ str  ┆ str   ┆ str               ┆ str               │
#     ╞══════╪═══════╪═══════════════════╪═══════════════════╡
#     │ road ┆ man   ┆ Alessandro Ballan ┆ Cadel Evans       │
#     │ itt  ┆ man   ┆ Bert Grabsch      ┆ Fabian Cancellara │
#     │ road ┆ woman ┆ Nicole Cooke      ┆ Tatiana Guderzo   │
#     │ itt  ┆ woman ┆ Amber Neben       ┆ Kristin Armstrong │
#     └──────┴───────┴───────────────────┴───────────────────┘
#
#     df.melt(
#         id_vars=["sex", "race"],
#         variable_name="year",
#         value_name="winner",
#     )
#
#     """
#     >>> shape: (8, 4)
#     ┌───────┬──────┬──────┬───────────────────┐
#     │ sex   ┆ race ┆ year ┆ winner            │
#     │ ---   ┆ ---  ┆ ---  ┆ ---               │
#     │ str   ┆ str  ┆ str  ┆ str               │
#     ╞═══════╪══════╪══════╪═══════════════════╡
#     │ man   ┆ road ┆ 2008 ┆ Alessandro Ballan │
#     │ man   ┆ itt  ┆ 2008 ┆ Bert Grabsch      │
#     │ woman ┆ road ┆ 2008 ┆ Nicole Cooke      │
#     │ woman ┆ itt  ┆ 2008 ┆ Amber Neben       │
#     │ man   ┆ road ┆ 2009 ┆ Cadel Evans       │
#     │ man   ┆ itt  ┆ 2009 ┆ Fabian Cancellara │
#     │ woman ┆ road ┆ 2009 ┆ Tatiana Guderzo   │
#     │ woman ┆ itt  ┆ 2009 ┆ Kristin Armstrong │
#     └───────┴──────┴──────┴───────────────────┘
#     """
#
#     (
#         df
#         .with_columns(cs.matches("\\d+").cast(pl.Categorical))
#         .melt(
#             id_vars=["sex", "race"],
#             variable_name="year",
#             value_name="winner",
#         )
#     )
#
#     """
#     >>> shape: (8, 4)
#     ┌───────┬──────┬──────┬───────────────────┐
#     │ sex   ┆ race ┆ year ┆ winner            │
#     │ ---   ┆ ---  ┆ ---  ┆ ---               │
#     │ str   ┆ str  ┆ str  ┆ cat               │
#     ╞═══════╪══════╪══════╪═══════════════════╡
#     │ man   ┆ road ┆ 2008 ┆ Alessandro Ballan │
#     │ man   ┆ itt  ┆ 2008 ┆ Bert Grabsch      │
#     │ woman ┆ road ┆ 2008 ┆ Nicole Cooke      │
#     │ woman ┆ itt  ┆ 2008 ┆ Amber Neben       │
#     │ man   ┆ road ┆ 2009 ┆ Alessandro Ballan │
#     │ man   ┆ itt  ┆ 2009 ┆ Bert Grabsch      │
#     │ woman ┆ road ┆ 2009 ┆ Nicole Cooke      │
#     │ woman ┆ itt  ┆ 2009 ┆ Amber Neben       │
#     └───────┴──────┴──────┴───────────────────┘
#     """
#
#     with pl.StringCache():
#         (
#             df
#             .with_columns(cs.matches("\\d+").cast(pl.Categorical))
#             .melt(
#                 id_vars=["sex", "race"],
#                 variable_name="year",
#                 value_name="winner",
#             )
#         )
#         """
#         >>> thread '<unnamed>' panicked at 'called `Option::unwrap()` on a `None` value', /home/runner/work/polars/polars/crates/polars-core/src/chunked_array/logical/categorical/builder.rs:112:42
#
#         ---------------------------------------------------------------------------
#         PanicException                            Traceback (most recent call last)
#         Cell In[17], line 2
#         1 pl.enable_string_cache(True)
#   ----> 2 print(
#         3     df
#         4     .with_columns(cs.matches("\\d+").cast(pl.Categorical))
#         5     .melt(
#         6         id_vars=["sex", "race"],
#         7         variable_name="year",
#         8         value_name="winner",
#         9     )
#         10 )
#
#         File ~/Notebooks/Engineering/2023-08 - CodinGame/.venv/lib/python3.11/site-packages/polars/dataframe/frame.py:1440, in DataFrame.__str__(self)
#         1439 def __str__(self) -> str:
#         -> 1440     return self._df.as_str()
#
#         PanicException: called `Option::unwrap()` on a `None` value
#         """
