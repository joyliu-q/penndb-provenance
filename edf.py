import typing as t
import pandas as pd
from error import PipelineError
from openai import OpenAI

from utils import OPENAI_API_KEY


class EDF(pd.DataFrame):
    """
    A pandas DataFrame that includes error tracking columns.
    Errors are stored as additional columns with prefix '_error_'.
    Global errors (not associated with rows) are stored in _global_errors.
    """

    _metadata = ["_error_prefix", "_global_errors", "openai_client", "_hash"]

    @property
    def _constructor(self):
        return EDF

    @property
    def _constructor_sliced(self):
        """
        This tells Pandas how to construct a "slice" of EDF, which is usually
        a Series-like object, from your subclassed DataFrame.
        """
        return pd.Series

    def __finalize__(self, other, method=None, **kwargs):
        """
        Ensures that custom attributes are copied over to the new EDF
        after certain operations (like groupby, concatenation, etc.).
        """
        super().__finalize__(other, method=method, **kwargs)
        self._error_prefix = getattr(other, "_error_prefix", "_error_")
        self._global_errors = getattr(other, "_global_errors", [])
        self.openai_client = getattr(other, "openai_client", None)
        self._hash = getattr(other, "_hash", self._calculate_hash())
        return self

    def __init__(self, *args, **kwargs):
        self._error_prefix = "_error_"
        self._global_errors = []
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

        super().__init__(*args, **kwargs)
        self._hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate a hash of the DataFrame content"""
        return pd.util.hash_pandas_object(self).sum()

    def has_changed(self) -> bool:
        """Check if DataFrame content has changed since last hash calculation"""
        current_hash = self._calculate_hash()
        return current_hash != self._hash

    def mark_unchanged(self):
        """Update hash to current state"""
        self._hash = self._calculate_hash()

    def register_natural_error(self, condition: str) -> "EDF":
        """
        Register errors using a 'must' condition in natural language.
        We interpret the condition as describing the *valid* state
        (e.g. "age must be >= 0"), then we flag rows that violate it.
        """
        category, description = PipelineError.classify_error(self.openai_client, condition)

        query_str = self._generate_error_query(condition)
        print(f"Query string: {query_str}")
        valid_idx = self.query(query_str).index
        invalid_idx = self.index.difference(valid_idx)  # these fail the condition

        if len(invalid_idx) > 0:
            column = None
            parts = query_str.split()
            if len(parts) > 0 and parts[0] in self.columns:
                column = parts[0]

            return self.register_error(
                row_idx=invalid_idx.tolist(),
                category=category,
                description=description,
                column=column,
            )

        return self

    def natural_query_errors(self, query: str) -> "EDF":
        """
        df.natural_query_errors("show me all age-related errors")
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")

        prompt = f"""Given this query: "{query}"
        And these error categories: {[e.name for e in PipelineError]}

        Should I filter by:
        1. Category? If yes, which one?
        2. Column? If yes, which column?

        Return in format: category:CATEGORY,column:COLUMN
        Use None if no filter needed."""

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a query parser."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        result = response.choices[0].message.content.strip()
        category_str = result.split("category:")[1].split(",")[0].strip()
        column_str = result.split("column:")[1].strip()

        if category_str == "None":
            category = None
        else:
            category = getattr(PipelineError, category_str, None)

        column = None if column_str == "None" else column_str

        return self.query_errors(category=category, column=column)

    def register_global_error(
        self, category: PipelineError, description: str, column: t.Optional[str] = None
    ) -> "EDF":
        """
        Register a global error not associated with any specific row.
        """
        df = self.copy()
        df._global_errors.append((int(category), description, column))
        return df

    def register_error(
        self,
        row_idx: t.Union[int, t.List[int]],
        category: PipelineError,
        description: str,
        column: t.Optional[str] = None,
    ) -> "EDF":
        """
        Register an error for specific rows in the DataFrame.
        If DataFrame is empty, registers as global error instead.
        """
        if self.empty:
            return self.register_global_error(category, description, column)

        df = self.copy()
        if isinstance(row_idx, int):
            row_idx = [row_idx]

        error_cat_col = f"{self._error_prefix}category"
        error_desc_col = f"{self._error_prefix}description"
        error_col_col = f"{self._error_prefix}column"

        if error_cat_col not in df.columns:
            df[error_cat_col] = pd.Series(dtype="Int64")
            df[error_desc_col] = pd.Series(dtype="string[python]")
            df[error_col_col] = pd.Series(dtype="string[python]")

        for r in row_idx:
            if r in df.index:
                df.at[r, error_cat_col] = int(category)
                df.at[r, error_desc_col] = str(description)
                if column:
                    df.at[r, error_col_col] = str(column)
                else:
                    df.at[r, error_col_col] = None

        return df

    def query_errors(
        self, category: t.Optional[PipelineError] = None, column: t.Optional[str] = None
    ) -> "EDF":
        """
        Query rows with errors and global errors, optionally filtered by category and/or column.
        Returns both row-specific and global errors.
        """
        # Start with an empty EDF
        result = EDF()

        # --- Gather any matching global errors ---
        matching_globals = []
        for err_cat, err_desc, err_col in self._global_errors:
            if (category is None or err_cat == int(category)) and (
                column is None or err_col == column
            ):
                matching_globals.append(
                    {
                        "error_type": "global",
                        "category": int(err_cat),
                        "description": err_desc,
                        "column": err_col,
                    }
                )

        if matching_globals:
            globals_df = pd.DataFrame(matching_globals)
            if not globals_df.empty:
                globals_df = globals_df.astype(
                    {
                        "error_type": "string[python]",
                        "category": "Int64",
                        "description": "string[python]",
                        "column": "string[python]",
                    }
                )
                result = pd.concat([result, globals_df])

        # --- Gather row-based errors ---
        error_cat_col = f"{self._error_prefix}category"
        error_col_col = f"{self._error_prefix}column"

        if error_cat_col in self.columns:
            mask = self[error_cat_col].notna()
            if category is not None:
                mask &= self[error_cat_col] == int(category)
            if column is not None:
                mask &= (self[error_col_col] == column) | (
                    self[error_col_col].isna() & (column is None)
                )

            row_errors = self[mask].copy()
            if not row_errors.empty:
                row_errors["error_type"] = "row"
                row_errors["error_type"] = row_errors["error_type"].astype("string[python]")
                result = pd.concat([result, row_errors])

        return result

    def has_errors(self) -> bool:
        """Check if DataFrame has any errors registered (global or row-specific)."""
        error_cat_col = f"{self._error_prefix}category"
        return bool(self._global_errors) or (
            error_cat_col in self.columns and self[error_cat_col].notna().any()
        )

    def clear_errors(self) -> "EDF":
        """Remove all error columns and global errors from DataFrame."""
        df = self.copy()
        error_cols = [col for col in df.columns if col.startswith(self._error_prefix)]
        df.drop(columns=error_cols, inplace=True, errors="ignore")
        df._global_errors = []
        return df

    def _generate_error_query(self, condition: str) -> str:
        """
        Use LLM to convert natural language condition (a valid rule) into a Pandas query
        that identifies rows that satisfy it.
        For "age must be >= 0", we want "age >= 0".
        """
        if not self.openai_client:
            # Fallback if no LLM client is available:
            # Simple heuristic-based parse for demonstration
            # e.g. "age must be >= 0" -> "age >= 0"
            # This obviously doesn't handle all cases or natural language well.
            if "must be" in condition:
                return condition.replace("must be", "")
            elif "must" in condition:
                # Handle "age must >= 0" pattern
                parts = condition.split("must")
                if len(parts) > 1:
                    col = parts[0].strip()
                    cond = parts[1].strip()
                    return f"{col} {cond}"
            # Return the original if we can't parse it
            raise ValueError(f"Failed to parse condition: {condition}")

        columns = list(self.columns)
        prompt = f"""Convert this requirement into a valid pandas query that selects only rows that meet it.
        Requirement: "{condition}"
        Available columns: {columns}
        Return just the query string. Example output: "age >= 0" or "score > 100".
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a pandas query generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=100,
        )

        query_str = response.choices[0].message.content.strip()
        if query_str.startswith('"') and query_str.endswith('"'):
            query_str = query_str[1:-1]
        return query_str


# Example
if __name__ == "__main__":
    df = EDF(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, -5],
        }
    )

    df_with_errors = df.register_error(
        row_idx=2,
        category=PipelineError.BAD_REQUEST,
        description="Age cannot be negative",
        column="age",
    )

    df_with_errors = df_with_errors.register_natural_error(
        "age must be greater than or equal to zero"
    )

    error_rows = df_with_errors.query_errors()

    average_age = df_with_errors.age.mean()
    grouped = df_with_errors.groupby("name").age.mean()

    print("Average age:", average_age)
    print("\nError rows:")
    print(error_rows)
    print("\nOriginal DataFrame with errors:")
    print(df_with_errors)

    age_errors = df_with_errors.natural_query_errors("show me all age-related errors")

    # add etl stage definition with following operations suported
    # - take nothing and output a df (load)
    # - take df and output that df (transform)
    # - take df and output a single value (fold)
    # - take multiple df and output a df (fold / aggregate)
