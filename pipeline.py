"""Pipeline Construction Tools"""

import functools
import logging
import typing as t
from graphviz import Digraph

from error import PipelineError
from edf import EDF
import pandas as pd

from abc import ABC

T = t.TypeVar("T")


# TODO: All stages belong to a pipeline, which thye must be registered to
class ETLStage(ABC):
    """Base class for all ETL stages."""

    def __init__(self, name: str):
        self.name = name
        self.dependencies: t.List[ETLStage] = []
        self.result = None

    def add_dependency(self, stage: 'ETLStage'):
        self.dependencies.append(stage)

    def get_dependencies(self) -> t.List['ETLStage']:
        return self.dependencies

    @property
    def has_run(self) -> bool:
        return self.result is not None


class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self.stages: t.Dict[str, ETLStage] = {}  # Changed to dict for name lookup
        self.last_stage: t.Optional[ETLStage] = None

    def add_stage(self, stage: ETLStage):
        """Add a stage to the pipeline without creating automatic dependencies"""
        self.stages[stage.name] = stage
        self.last_stage = stage

    def depends_on(self, *stage_names: str):
        """Decorator to specify stage dependencies"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Store dependencies to be resolved when the stage is created
            wrapper._dependencies = stage_names
            return wrapper
        return decorator

    def _create_stage(self, func: t.Callable, stage_class: t.Type[ETLStage]) -> ETLStage:
        """Helper to create a stage with dependencies"""
        stage = stage_class(func.__name__, func)
        
        # Add explicit dependencies if specified
        if hasattr(func, '_dependencies'):
            for dep_name in func._dependencies:
                if dep_name not in self.stages:
                    raise ValueError(f"Dependency '{dep_name}' not found for stage '{func.__name__}'")
                stage.add_dependency(self.stages[dep_name])
        
        return stage

    def extract(self, func: t.Callable[[], EDF]) -> t.Callable[[], EDF]:
        """Decorator for creating an ExtractStage."""
        stage = self._create_stage(func, ExtractStage)
        self.add_stage(stage)

        def wrapper() -> EDF:
            return stage.execute()
        return wrapper

    def transform(self, func: t.Callable[[EDF], EDF]) -> t.Callable[[EDF], EDF]:
        """Decorator for creating a TransformStage."""
        stage = self._create_stage(func, TransformStage)
        self.add_stage(stage)

        def wrapper(df: EDF) -> EDF:
            return stage.execute(df)

        return wrapper

    def fold(self, func: t.Callable[[EDF], T]) -> t.Callable[[EDF], T]:
        """Decorator for creating a FoldStage."""
        stage = self._create_stage(func, FoldStage)
        self.add_stage(stage)

        def wrapper(df: EDF) -> T:
            return stage.execute(df)

        return wrapper

    def aggregate(self, func: t.Callable[[t.List[EDF]], EDF]) -> t.Callable[[t.List[EDF]], EDF]:
        """Decorator for creating an AggregateStage."""
        stage = self._create_stage(func, AggregateStage)
        self.add_stage(stage)

        def wrapper(dfs: t.List[EDF]) -> EDF:
            return stage.execute(dfs)

        return wrapper

    def run(self) -> t.Dict[str, t.Any]:
        """Execute the pipeline in dependency order"""
        results = {}
        executed = set()

        def execute_stage(stage: ETLStage):
            if stage.name in executed:
                return stage.result

            dep_results = []
            for dep in stage.dependencies:
                dep_result = execute_stage(dep)
                dep_results.append(dep_result)

            if isinstance(stage, ExtractStage):
                if dep_results:
                    raise ValueError(f"Extract stage {stage.name} should not have dependencies")
                stage.result = stage.execute()
            elif isinstance(stage, TransformStage):
                if len(dep_results) != 1:
                    raise ValueError(f"Transform stage {stage.name} expects exactly one dependency, got {len(dep_results)}")
                stage.result = stage.execute(dep_results[0])
            elif isinstance(stage, FoldStage):
                if len(dep_results) != 1:
                    raise ValueError(f"Fold stage {stage.name} expects exactly one dependency, got {len(dep_results)}")
                stage.result = stage.execute(dep_results[0])
            elif isinstance(stage, AggregateStage):
                stage.result = stage.execute(dep_results)  # Aggregate can take multiple dependencies

            executed.add(stage.name)
            results[stage.name] = stage.result
            return stage.result

        for stage in self.stages.values():
            execute_stage(stage)

        return results

    def visualize(self, filename: t.Optional[str] = None) -> None:
        """
        Visualize the pipeline as a DAG using graphviz.
        
        Args:
            filename: Name of the output file (without extension)
        """
        dot = Digraph(comment=f'Pipeline: {self.name}')
        dot.attr(rankdir='LR')
        if filename is None:
            filename = self.name

        for stage_name, stage in self.stages.items():
            color = {
                ExtractStage: 'lightblue',
                TransformStage: 'lightgreen',
                FoldStage: 'lightyellow',
                AggregateStage: 'lightpink'
            }.get(type(stage), 'white')
            
            dot.node(stage_name, stage_name, style='filled', fillcolor=color)

        for stage_name, stage in self.stages.items():
            for dep in stage.get_dependencies():
                dot.edge(dep.name, stage_name)

        dot.render(filename, view=True, format='svg')


class ExtractStage(ETLStage):
    def __init__(self, name: str, loader: t.Callable[[], EDF]):
        super().__init__(name)
        self._loader = loader

    def execute(self) -> EDF:
        return self._loader()


class TransformStage(ETLStage):
    def __init__(self, name: str, transformer: t.Callable[[EDF], EDF]):
        super().__init__(name)
        self._transformer = transformer

    def execute(self, df: EDF) -> EDF:
        return self._transformer(df)


class FoldStage(ETLStage, t.Generic[T]):
    """Stage that reduces a DataFrame to a single value."""

    def __init__(self, name: str, folder: t.Callable[[EDF], T]):
        super().__init__(name)
        self._folder = folder

    def execute(self, df: EDF) -> T:
        return self._folder(df)


class AggregateStage(ETLStage):
    """Stage that combines multiple DataFrames into one."""

    def __init__(self, name: str, aggregator: t.Callable[[t.List[EDF]], EDF]):
        super().__init__(name)
        self._aggregator = aggregator

    def execute(self, dfs: t.List[EDF]) -> EDF:
        return self._aggregator(dfs)


pipeline = Pipeline("My ETL Pipeline")


@pipeline.extract
def load_data() -> EDF:
    return EDF(pd.read_csv("data.csv"))


@pipeline.transform
def clean_data(df: EDF) -> EDF:
    return df.register_natural_error("age must be >= 0")


@pipeline.fold
def calc_average(df: EDF) -> float:
    return df.age.mean()


@pipeline.aggregate
def combine_dfs(dfs: t.List[EDF]) -> EDF:
    return pd.concat(dfs, ignore_index=True)


class RowLevelPipelineError(Exception):
    """
    Raise this exception when you want to register an error for specific row(s)
    rather than globally.
    """

    def __init__(
        self,
        row_idx: t.Union[int, t.List[int]],
        category: PipelineError,
        description: str,
        column: t.Optional[str] = None,
    ):
        super().__init__(description)
        self.row_idx = row_idx
        self.category = category
        self.column = column


def pipeline_error_handler(
    stage_name: str,
    error_classes: t.Union[type, t.Tuple[type, ...]],
    default_category: PipelineError = PipelineError.BAD_REQUEST,
):
    """
    A decorator that:
      - Expects the wrapped function's first argument to be an EDF (our target DF).
      - Catches ONLY exceptions of the specified types (error_classes).
      - If it's a RowLevelPipelineError, registers row-based error(s).
      - Otherwise, registers a global error with 'default_category'.
      - Returns the EDF (with errors) after catching, or re-raises if the exception type is not matched.

    :param stage_name: Identifies which pipeline stage is being decorated (for logging/errors).
    :param error_classes: Exception class or tuple of classes to catch.
    :param default_category: The fallback category for non-row-level errors.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(edf: EDF, *args, **kwargs) -> EDF:
            try:
                return func(edf, *args, **kwargs)
            except error_classes as e:
                logging.exception(f"Error in stage '{stage_name}': {str(e)}")

                if isinstance(e, RowLevelPipelineError):
                    # --- Row-level registration ---
                    edf_with_error = edf.register_error(
                        row_idx=e.row_idx,
                        category=e.category,
                        description=str(e),
                        column=e.column,
                    )
                    return edf_with_error
                else:
                    # --- Fallback global registration ---
                    edf_with_error = edf.register_global_error(
                        category=default_category,
                        description=f"Error in stage '{stage_name}': {str(e)}",
                    )
                    return edf_with_error

        return wrapper

    return decorator
