import os
import sys
import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)


def _call_site(watched_files: list[str]) -> str:
    frame = sys._getframe(1)
    while frame is not None:
        base = frame.f_code.co_filename.rsplit("/", 1)[-1]
        if base in watched_files:
            return f"{base}:{frame.f_lineno}"
        frame = frame.f_back
    return "-"


def _dtypes(x) -> str:
    from torch.utils._pytree import tree_flatten
    flat, _ = tree_flatten(x)
    return ",".join(str(t.dtype)[6:] for t in flat if isinstance(t, torch.Tensor)) or "-"


class ModuleStack:
    def __init__(self, root: nn.Module, watched_files: list[str]):
        self._fqn = {id(m): (name or "model") for name, m in root.named_modules()}
        self.stack: list[str] = []
        self.rows: list[tuple] = []
        self.watched_files = watched_files

    @property
    def current(self) -> str:
        return self.stack[-1] if self.stack else "global"

    def _pre(self, mod, inp):
        self.stack.append(self._fqn.get(id(mod), type(mod).__name__))

    def _post(self, mod, inp, out):
        name = self.stack.pop() if self.stack else "global"
        params_and_buffers = []
        params_and_buffers += [f"{n}:{str(p.dtype)[6:]}" for n, p in mod.named_parameters(recurse=False)]
        params_and_buffers += [f"{n}:{str(b.dtype)[6:]}" for n, b in mod.named_buffers(recurse=False)]
        params_and_buffers = ",".join(params_and_buffers)
   
        self.rows.append(
            (name, type(mod).__name__, _call_site(self.watched_files), params_and_buffers or "-",
                _dtypes(inp), _dtypes(out))
        )

    def __enter__(self):
        self._handles = [
            register_module_forward_pre_hook(self._pre),
            register_module_forward_hook(self._post),
        ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self.stack.clear()
        self.rows.clear()


class DtypeTracer(TorchFunctionMode):
    def __init__(self, mstack: ModuleStack, watched_files: list[str], skip_ops: list[str]):
        super().__init__()
        self.mstack = mstack
        self.rows: list[tuple] = []
        self.watched_files = watched_files
        self.skip_ops = skip_ops

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        op = getattr(func, "__name__", str(func))
        if op in self.skip_ops or not isinstance(out, torch.Tensor):
            return out
        in_dtypes = [str(a.dtype)[6:] for a in args if isinstance(a, torch.Tensor)]
        self.rows.append(
            (self.mstack.current, op, _call_site(self.watched_files),
             ",".join(in_dtypes) or "-", str(out.dtype)[6:])
        )
        return out


def save_rows_markdown(mstack_rows: list[tuple], tracer_rows: list[tuple], path: str):
    # Helper to format a row
    def fmt_row(values, col_widths):
        return "| " + " | ".join(
            str(v).replace("\n", " ").ljust(w) for v, w in zip(values, col_widths)
        ) + " |\n"

    # Helper to format separator
    def fmt_sep(col_widths):
        return "|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n"

    # create folder
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # ============ write mstack_rows ============
    headers = ["FQN", "Module", "Callsite", "Params and Buffers", "Input dtypes", "Output dtypes"]
    col_widths = [len(h) for h in headers]
    for name, module, callsite, params, in_dtypes, out_dtypes in mstack_rows:
        col_widths[0] = max(col_widths[0], len(str(name)))
        col_widths[1] = max(col_widths[1], len(str(module)))
        col_widths[2] = max(col_widths[2], len(str(callsite)))
        col_widths[3] = max(col_widths[3], len(str(params)))
        col_widths[4] = max(col_widths[4], len(str(in_dtypes)))
        col_widths[5] = max(col_widths[5], len(str(out_dtypes)))

    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Module\n')
        f.write(fmt_row(headers, col_widths))
        f.write(fmt_sep(col_widths))
        for name, module, callsite, params, in_dtypes, out_dtypes in mstack_rows:
            values = [
                name,
                module,
                callsite,
                params,
                str(in_dtypes).replace("\n", " "),
                str(out_dtypes).replace("\n", " "),
            ]
            f.write(fmt_row(values, col_widths))
        f.write('\n\n')

    # ============ write tracer_rows ============
    headers = ["FQN", "Op", "Callsite", "Input dtypes", "Output dtype"]
    col_widths = [len(h) for h in headers]
    for module, op, callsite, in_dtypes, out_dtype in tracer_rows:
        col_widths[0] = max(col_widths[0], len(str(module)))
        col_widths[1] = max(col_widths[1], len(str(op)))
        col_widths[2] = max(col_widths[2], len(str(callsite)))
        in_dtypes_str = str(in_dtypes).replace("\n", " ")
        col_widths[3] = max(col_widths[3], len(in_dtypes_str))
        col_widths[4] = max(col_widths[4], len(str(out_dtype)))

    with open(path, 'a', encoding='utf-8') as f:
        f.write('# Op\n')
        f.write(fmt_row(headers, col_widths))
        f.write(fmt_sep(col_widths))
        for module, op, callsite, in_dtypes, out_dtype in tracer_rows:
            values = [
                module,
                op,
                callsite,
                str(in_dtypes).replace("\n", " "),
                out_dtype,
            ]
            f.write(fmt_row(values, col_widths))


def debug_dump_dtype(model, fwd_kwargs, dump_path, watched_files, skip_ops):
    # dump dtype of all module params, input and output,
    # and all op (any torch.* or Tensor.*) input and output
    with ModuleStack(model, watched_files) as mstack, DtypeTracer(mstack, watched_files, skip_ops) as tracer:
        model(**fwd_kwargs)
        if torch.distributed.get_rank() == 0:
            save_rows_markdown(mstack.rows, tracer.rows, dump_path)
            print(f'[Debug dump] Dumped dtype to {dump_path}. Exit now.')
        sys.exit(0)