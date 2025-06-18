# main.py - Analizador de Complejidad Temporal
import sys
import os
from tkinter import messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    missing_deps = []

    try:
        import customtkinter
    except ImportError:
        missing_deps.append("customtkinter")

    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        import reportlab
    except ImportError:
        missing_deps.append("reportlab")

    try:
        import ply
    except ImportError:
        missing_deps.append("ply")

    if missing_deps:
        deps_str = ", ".join(missing_deps)
        error_msg = f"Dependencias faltantes: {deps_str}\nInstalar con: pip install {' '.join(missing_deps)}"
        print(error_msg)
        try:
            messagebox.showerror("Dependencias Faltantes", error_msg)
        except:
            pass
        return False

    return True


def main():
    if not check_dependencies():
        sys.exit(1)

    try:
        from gui import ComplexityAnalyzerGUI
        app = ComplexityAnalyzerGUI()
        app.run()

    except ImportError as e:
        error_msg = f"Error al importar módulos: {str(e)}"
        print(error_msg)
        try:
            messagebox.showerror("Error de Importación", error_msg)
        except:
            pass
        sys.exit(1)

    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        print(error_msg)
        try:
            messagebox.showerror("Error Fatal", error_msg)
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()