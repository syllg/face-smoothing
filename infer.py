import os
import sys
import traceback


def _bootstrap_import_path():
    """Set import path for development mode only."""
    if getattr(sys, "frozen", False):
        return

    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)


def run():
    _bootstrap_import_path()
    from face_smoothing.main import main
    main()


if __name__ == "__main__":
    try:
        run()
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        raise