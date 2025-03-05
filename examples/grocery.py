from edf import EDF
from error import PipelineError
from examples.error_factory import rl_fetch
from pipeline import Pipeline, RowLevelPipelineError, pipeline_error_handler
from tqdm import tqdm
from openai import OpenAI
from langchain.prompts import PromptTemplate
from utils import OPENAI_API_KEY, get_llm
from agent import SearchSerperAgent

import requests
import pandas as pd
import typing as t
import os

pipeline = Pipeline(
    "My ETL Pipeline",
    redis_url="redis://localhost:6379/0"
)

client = OpenAI(api_key=OPENAI_API_KEY)


@pipeline.extract
def load_internal_data() -> EDF:
    try:
        file_path = os.path.join(os.path.dirname(__file__), "data", "WMT_Grocery_202209.csv")
        df = pd.read_csv(file_path, low_memory=False)
        # convert schema to product_name -> product, price_current -> price
        df = df.rename(
            columns={
                "PRODUCT_NAME": "product",
                "PRICE_CURRENT": "price",
                "CATEGORY": "category",
            }
        )
        edf = EDF(df)
        return EDF(df)
    except FileNotFoundError as e:
        edf = EDF()
        print(f"File Not Found when reading grocery data: {str(e)}")
        edf.register_global_error(
            category=PipelineError.NOT_FOUND,
            description=f"File Not Found when reading grocery data: {str(e)}",
            column="load_internal_data",
        )
        return edf
    except pd.errors.EmptyDataError as e:
        edf = EDF()
        edf.register_global_error(
            category=PipelineError.NOT_FOUND,
            description=f"No results found when loading internal data: {str(e)}",
            column="load_internal_data",
        )
        return edf


@pipeline.extract
def fetch_external_data() -> EDF:
    """
    Fetch products from Fake Store API and return the updated DataFrame.
    """
    try:
        # response = requests.get('https://fakestoreapi.com/products')
        response = rl_fetch("https://dummyjson.com/c/b8b7-1d1f-4b4b-8612")
        if response.status_code == 200:
            competitor_products = response.json()

            competitor_df = pd.DataFrame(competitor_products)

            return EDF(competitor_df)
        else:
            raise Exception(f"Failed to fetch data: Status code {response.status_code}")

    except Exception as e:
        df = EDF()
        df = df.register_global_error(
            category=PipelineError.SERVICE_UNAVAILABLE,
            description=f"Error fetching from Fake Store API: {str(e)}",
            column="fetch_prices",
        )
        return df


@pipeline.transform
@pipeline_error_handler(
    stage_name="clean_external_data",
    error_classes=(KeyError, ValueError),
    default_category=PipelineError.EXTERNAL_ERROR,
)
@pipeline.depends_on("fetch_external_data")
def clean_external_data(competitor_df: EDF) -> EDF:
    """
    Clean external data:
    - Renames columns for schema resolution.
    - Filters out missing/invalid data.
    - Registers row-specific errors for missing critical values.
    """
    # Ensure input EDF exists
    source = "apistore"
    if competitor_df.empty:
        raise ValueError(f"Input dataframe for '{source}' is empty.")

    # --- Schema resolution ---
    competitor_df = competitor_df.rename(columns={"title": "product"})
    # -- Prefix column with competitor source name (e.g. Walmart)
    competitor_df = competitor_df.rename(columns=lambda x: f"{source}_{x}")

    # --- Data Cleaning ---
    required_cols = [
        f"{source}_product",
        f"{source}_price",
        f"{source}_category",
        f"{source}_description",
    ]

    for col in required_cols:
        if col not in competitor_df.columns:
            raise KeyError(f"Missing expected column: {col}")

        missing_rows = competitor_df[competitor_df[col].isna()].index.tolist()
        if missing_rows:
            raise RowLevelPipelineError(
                row_idx=missing_rows,
                category=PipelineError.EXTERNAL_ERROR,
                description=f"Missing values detected in '{col}'",
                column=col,
            )

    return competitor_df


@pipeline.aggregate
@pipeline_error_handler(
    stage_name="merge_data",
    error_classes=(ValueError, RowLevelPipelineError),
    default_category=PipelineError.BAD_RESPONSE,
)
@pipeline.depends_on("load_internal_data", "clean_external_data")
def merge_data(dfs: t.List[EDF]) -> EDF:
    internal_df, external_df = dfs
    limit_rows = 100
    external_source = "apistore"
    try:
        merged_df = EDF()

        external_list = [
            {
                "id": int(row.name),
                "product_name": row[f"{external_source}_product"],
                # "description": row[f'{external_source}_description'],
            }
            for _, row in external_df.iterrows()
        ]

        limited_internal_df = internal_df.head(limit_rows)

        print(external_list)

        for idx, internal_row in tqdm(
            limited_internal_df.iterrows(),
            total=len(limited_internal_df),
            desc="Merging Data",
        ):

            internal_product = internal_row["product"]

            system_prompt = "You are a helpful assistant. You will be given a product name and a list of products. Return ONLY the ID number of the best matching product, or -1 if no good match exists."
            user_prompt = f"""Compare this product name: "{internal_product}" with the following products and return the id of the best match. Return ONLY the ID number, nothing else.

Products:
{external_list}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=3,
                    temperature=0.0,
                )

                match_index = int(response.choices[0].message.content.strip())
                print(f"Match index for '{internal_product}': {match_index}")

            except Exception as e:
                error_msg = str(e)
                if "insufficient_quota" in error_msg or "exceeded your current quota" in error_msg:
                    # Return partial results if we have any
                    if not merged_df.empty:
                        return merged_df
                    # Otherwise raise as a quota error
                    raise RowLevelPipelineError(
                        row_idx=idx,
                        category=PipelineError.LIMIT_EXCEEDED,
                        description=f"OpenAI API quota exceeded: {error_msg}",
                        column="merge_data",
                    )
                else:
                    raise RowLevelPipelineError(
                        row_idx=idx,
                        category=PipelineError.BAD_RESPONSE,
                        description=f"Error during match comparison: {error_msg}",
                        column="merge_data",
                    )

            if match_index >= 0:
                try:
                    external_row = external_df.iloc[match_index]
                    merged_row = {
                        **internal_row.to_dict(),
                        **external_row.to_dict(),
                    }
                    print(f"Adding merged row: {merged_row}")
                    merged_df = pd.concat([merged_df, EDF([merged_row])], ignore_index=True)
                except Exception as e:
                    print(f"Error adding merged row: {str(e)}")
                    raise RowLevelPipelineError(
                        row_idx=idx,
                        category=PipelineError.BAD_REQUEST,
                        description=f"Error adding merged row: {str(e)}",
                        column="merge_data",
                    )

        return merged_df

    except Exception as e:
        print(f"Error during data merging: {str(e)}")
        edf = EDF()
        edf.register_global_error(
            category=PipelineError.BAD_REQUEST,
            description=f"Uncaught error during data merging: {str(e)}",
            column="merge_data",
        )
        return edf

@pipeline.transform
@pipeline_error_handler(
    stage_name="merge_data",
    error_classes=(requests.exceptions.RequestException, ValueError, KeyError, RowLevelPipelineError),
    default_category=PipelineError.EXTERNAL_ERROR,
)
@pipeline.depends_on("load_internal_data")
def enrich_internal_data_with_web(product_data_df: EDF) -> EDF:
    """
    Looks up product name and fetches additional information from the web
    """
    output_df = product_data_df.copy()
    for idx, row in tqdm(output_df.iterrows(), total=len(output_df), desc="Enriching with Web"):
        try:
            res = SearchSerperAgent.run(row["product"])
            output_df.at[idx, "web_search_data"] = res
        # TODO: build this into the agent side, but we need to handle how to pass in row_idx to refactor things a bit
        except requests.exceptions.Timeout as e:
            raise RowLevelPipelineError(
                row_idx=idx,
                category=PipelineError.SERVICE_UNAVAILABLE,
                description=f"Serper API request timed out. The service may be experiencing high load: {str(e)}",
                column="web_search_data"
            )
        except requests.exceptions.RequestException as e:
            if "Unauthorized" in str(e) or "403" in str(e):
                raise RowLevelPipelineError(
                    row_idx=idx,
                    category=PipelineError.UNAUTHORIZED,
                    description=f"Unauthorized access to Serper API. Please check API credentials: {str(e)}",
                    column="web_search_data"
                )
            raise RowLevelPipelineError(
                row_idx=idx,
                category=PipelineError.SERVICE_UNAVAILABLE,
                description=f"Error accessing Serper API: {str(e)}",
                column="web_search_data"
            )
    return output_df

@pipeline.fold
@pipeline_error_handler(
    stage_name="compare_products",
    error_classes=(KeyError, ValueError, TypeError),  # Only catch expected errors
    default_category=PipelineError.BAD_RESPONSE,
)
@pipeline.depends_on("merge_data")
def compare_products(merged_df: EDF) -> EDF:
    """
    Compare products in the merged DataFrame and classify them based on:
    - Whether our price is definitely better
    - Whether our product name sounds nicer
    """

    if merged_df.empty:
        raise ValueError("Input merged_df is empty.")

    results = []

    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Comparing Products"):

        try:
            our_price = row.get("price", None)
            competitor_price = row.get("apistore_price", None)

            our_product = row.get("product", "").strip()
            competitor_product = row.get("apistore_product", "").strip()

            price_better = None
            if pd.notna(our_price) and pd.notna(competitor_price):
                price_better = our_price < competitor_price

            product_sounds_nicer = None
            if our_product and competitor_product:
                product_sounds_nicer = len(our_product) > len(
                    competitor_product
                )  # A basic heuristic

            results.append(
                {
                    **row.to_dict(),
                    "price_better": price_better,
                    "product_sounds_nicer": product_sounds_nicer,
                }
            )

        except KeyError as e:
            raise RowLevelPipelineError(
                row_idx=idx,
                category=PipelineError.SERVICE_UNAVAILABLE,
                description=f"Missing required key in row: {str(e)}",
                column=str(e),
            ) from e

        except (TypeError, ValueError) as e:
            raise RowLevelPipelineError(
                row_idx=idx,
                category=PipelineError.BAD_RESPONSE,
                description=f"Invalid data type in row: {str(e)}",
                column="compare_products",
            ) from e

    return EDF(results)


@pipeline.fold
@pipeline_error_handler(
    stage_name="assess_aggregate_relationships",
    error_classes=(ValueError, RowLevelPipelineError),
    default_category=PipelineError.BAD_RESPONSE,
)
@pipeline.depends_on("compare_products")
def assess_aggregate_relationships(dataframe: EDF) -> EDF:
    llm = get_llm(0.7)
    joined_entry_list = ", ".join(
        [f"({', '.join([repr(value) for value in row])})" for row in dataframe.values]
    )
    descriptor = """Given the following products from competitors, and data about prices,
    categories, ratings, generate a detailed competitor product analysis of the products
    and categories, giving suggestions about our strategy"""

    headers = ", ".join(dataframe.columns)
    template = PromptTemplate(
        template="""Here is a list of objects with data for the following attributes:
        {headers}
        Objects:
        {joined_entry_list}
        Generate a summary analyzing these items holistically for: {descriptor}""",
        input_variables=["headers", "joined_entry_list", "descriptor"],
    )
    final_prompt = template.format(
        headers=headers, joined_entry_list=joined_entry_list, descriptor=descriptor
    )
    output = llm.predict(final_prompt)
    return EDF({"competitor_product_analysis": [output]})


results = pipeline.run()
pipeline.visualize()
final_df = results["assess_aggregate_relationships"]
print("=" * 10 + "Final DataFrame" + "=" * 10)
print(final_df.to_string(index=False))
print("=" * 10 + "Error Patterns" + "=" * 10)
print(pipeline.identify_error_patterns())
# final_df = final_df.register_natural_error(
#     """DATA QUALITY ERROR: if the category is a 'Hummus, Dips, & Salsa' then the
#     apistore_product must also be an actual 'Hummus, Dips, & Salsa'""",
# )
# final_df.query_errors()
