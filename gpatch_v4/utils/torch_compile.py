import functools
import inspect
import torch


def compile_dynamic_dims(
    dynamic_dims: dict[str, int | list[int] | tuple[int, ...]] | None = None,
    **compile_kwargs,
):
    """
    Compile a function with dynamic dimensions.

    Args:
        dynamic_dims: A dictionary of dynamic dimensions.
        **compile_kwargs: Keyword arguments for torch.compile.

    Returns:
        A decorated function.

    Example:
        # The function forward will be compiled with dynamic dimensions 1 (e.g., seq_len) of hidden_states and input_ids.
        @compile_dynamic_dims(dynamic_dims={"hidden_states": 1, "input_ids": 1}, fullgraph=True)
        def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
            ...
    
    NOTE: Do NOT compile FSDP hooks, such as the __call__ method of a fully_sharded FSDP module.
    """
    dynamic_dims = dict() if dynamic_dims is None else dict(dynamic_dims)
    def decorate(fn):
        parameter_indices = {
            name: index
            for index, name in enumerate(inspect.signature(fn).parameters)
        }
        compiled_fn = torch.compile(fn, **compile_kwargs)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if torch.compiler.is_compiling():
                return fn(*args, **kwargs)

            for name, dims in dynamic_dims.items():
                tensor = (
                    kwargs[name]
                    if name in kwargs
                    else args[parameter_indices[name]]
                )
                assert isinstance(tensor, torch.Tensor), (
                    f"{fn.__qualname__} argument {name} must be a Tensor, got {type(tensor)}"
                )
                torch._dynamo.mark_dynamic(tensor, dims)
            return compiled_fn(*args, **kwargs)
        return wrapper
    return decorate
