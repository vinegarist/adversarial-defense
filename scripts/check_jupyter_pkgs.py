import importlib
try:
    import importlib.metadata as md
except Exception:
    import importlib_metadata as md

pkgs = ['ipykernel','jupyter_client','tornado','jupyter_core','notebook','jupyter_server']
for p in pkgs:
    try:
        v = md.version(p)
    except Exception:
        try:
            m = importlib.import_module(p)
            v = getattr(m, '__version__', 'unknown')
        except Exception:
            v = 'NOT_INSTALLED'
    print(f"{p}: {v}")
