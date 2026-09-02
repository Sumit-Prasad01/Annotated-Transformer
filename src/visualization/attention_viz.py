import pandas as pd
import altair as alt
from utils.logger import logger
from utils.custom_exception import CustomException


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    """
    Convert attention weight matrix into DataFrame format for Altair heatmap plotting.
    """
    try:
        return pd.DataFrame(
            [
                (
                    r,
                    c,
                    float(m[r, c]),
                    "%.3f" % m[r, c],
                    row_tokens[r],
                    col_tokens[c],
                )
                for r in range(m.shape[0])
                for c in range(m.shape[1])
                if r < max_row and c < max_col
            ],
            columns=["row", "column", "value", "label", "row_token", "col_token"],
        )
    except Exception as e:
        logger.error("Error converting matrix to DataFrame.")
        raise CustomException("Failed to convert matrix to DataFrame", e)


def attn_map(df, height=250, width=250):
    """
    Generate Altair heatmap chart for multi-head attention matrix visualization.
    """
    try:
        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X("col_token:N", axis=alt.Axis(title=""), sort=None),
                y=alt.Y("row_token:N", axis=alt.Axis(title=""), sort=None),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["row_token", "col_token", "value"],
            )
            .properties(height=height, width=width)
        )
        return chart
    except Exception as e:
        logger.error("Error building attention heatmap chart.")
        raise CustomException("Failed to build attention heatmap chart", e)
