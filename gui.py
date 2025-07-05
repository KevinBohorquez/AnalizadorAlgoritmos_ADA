# gui.py - Interfaz Gráfica para el Analizador de Complejidad Temporal (MODIFICADA)
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkinter as FigureCanvasTkAgg

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from time_complexity_analizer import TimeComplexityAnalyzer


class ComplexityAnalyzerGUI:
    """Interfaz gráfica principal para el analizador de complejidad temporal"""

    def __init__(self):
        # Configurar tema
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Crear ventana principal
        self.root = ctk.CTk()
        self.root.title("Analizador de Complejidad Temporal")
        self.root.geometry("1400x900")

        # Variables
        self.analyzer = TimeComplexityAnalyzer()
        self.current_result = None
        self.current_code = ""

        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de usuario"""

        # Frame principal
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Título
        title_label = ctk.CTkLabel(
            main_frame,
            text="🔬 Analizador de Complejidad Temporal",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=(20, 30))

        # Frame horizontal para dividir entrada y resultados
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # ============= PANEL IZQUIERDO (ENTRADA) =============
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=20)

        # Etiqueta de entrada
        input_label = ctk.CTkLabel(left_panel, text="📝 Código a Analizar:", font=ctk.CTkFont(size=18, weight="bold"))
        input_label.pack(pady=(20, 15), padx=20, anchor="w")

        # Frame para botones de entrada (PRIMERA FILA)
        input_buttons_frame1 = ctk.CTkFrame(left_panel)
        input_buttons_frame1.pack(fill="x", padx=20, pady=(0, 8))

        self.load_file_btn = ctk.CTkButton(
            input_buttons_frame1,
            text="📁 Cargar Archivo",
            command=self.load_file,
            width=140,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.load_file_btn.pack(side="left", padx=(15, 8), pady=12)

        self.clear_btn = ctk.CTkButton(
            input_buttons_frame1,
            text="🗑️ Limpiar",
            command=self.clear_text,
            width=120,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="gray",
            hover_color="darkgray"
        )
        self.clear_btn.pack(side="left", padx=8, pady=12)

        self.analyze_btn = ctk.CTkButton(
            input_buttons_frame1,
            text="🔍 Analizar",
            command=self.analyze_code,
            width=140,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.analyze_btn.pack(side="left", padx=8, pady=12)

        # Frame para botones de entrada (SEGUNDA FILA)
        input_buttons_frame2 = ctk.CTkFrame(left_panel)
        input_buttons_frame2.pack(fill="x", padx=20, pady=(0, 15))

        self.export_report_btn = ctk.CTkButton(
            input_buttons_frame2,
            text="📋 PDF",
            command=self.export_report,
            width=100,
            height=35,
            state="disabled",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="blue",
            hover_color="darkblue"
        )
        self.export_report_btn.pack(side="left", padx=(15, 5), pady=12)

        self.export_graph_btn = ctk.CTkButton(
            input_buttons_frame2,
            text="📊 Ver Gráfica",
            command=self.show_graph,
            width=120,
            height=35,
            state="disabled",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="purple",
            hover_color="darkmagenta"
        )
        self.export_graph_btn.pack(side="left", padx=5, pady=12)

        # NUEVO BOTÓN: Comparar Algoritmos
        self.compare_algorithms_btn = ctk.CTkButton(
            input_buttons_frame2,
            text="⚖️ Comparar",
            command=self.open_comparison_window,
            width=130,
            height=35,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="darkorange",
            hover_color="chocolate"
        )
        self.compare_algorithms_btn.pack(side="left", padx=5, pady=12)

        # Área de texto para código
        self.code_text = ctk.CTkTextbox(
            left_panel,
            height=400,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="none"
        )
        self.code_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # ============= PANEL DERECHO (RESULTADOS) =============
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)

        # Etiqueta de resultados
        results_label = ctk.CTkLabel(right_panel, text="📊 Resultados del Análisis:",
                                     font=ctk.CTkFont(size=18, weight="bold"))
        results_label.pack(pady=(20, 15), padx=20, anchor="w")

        # Frame para métricas principales
        metrics_frame = ctk.CTkFrame(right_panel)
        metrics_frame.pack(fill="x", padx=20, pady=(0, 15))

        # Complejidad
        self.complexity_label = ctk.CTkLabel(
            metrics_frame,
            text="Complejidad: ⏳ No analizado",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="orange"
        )
        self.complexity_label.pack(pady=(15, 8), padx=20, anchor="w")

        # Función tiempo
        self.function_label = ctk.CTkLabel(
            metrics_frame,
            text="Función T: ⏳ No analizado",
            font=ctk.CTkFont(size=12),
            text_color="lightblue"
        )
        self.function_label.pack(pady=(0, 15), padx=20, anchor="w")

        # Área de texto para análisis detallado
        analysis_label = ctk.CTkLabel(right_panel, text="📄 Reporte Completo:", font=ctk.CTkFont(size=14, weight="bold"))
        analysis_label.pack(pady=(0, 10), padx=20, anchor="w")

        self.results_text = ctk.CTkTextbox(
            right_panel,
            height=250,
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.results_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Código de ejemplo
        self.load_example_code()

    def load_example_code(self):
        """Carga código de ejemplo"""
        example_code = """for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
        if(array[i] == array[j]) {
            return true;
        }
        sum = sum + array[i] * array[j];
    }
}"""
        self.code_text.insert("1.0", example_code)

    def load_file(self):
        """Carga un archivo de texto"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de código",
            filetypes=[
                ("Archivos de texto", "*.txt"),
                ("Archivos C++", "*.cpp"),
                ("Archivos C", "*.c"),
                ("Todos los archivos", "*.*")
            ]
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.code_text.delete("1.0", "end")
                    self.code_text.insert("1.0", content)
                messagebox.showinfo("✅ Éxito", f"Archivo cargado correctamente:\n{os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("❌ Error", f"Error al cargar el archivo:\n{str(e)}")

    def clear_text(self):
        """Limpia el área de texto y resultados"""
        self.code_text.delete("1.0", "end")
        self.results_text.delete("1.0", "end")
        self.complexity_label.configure(text="Complejidad: ⏳ No analizado", text_color="orange")
        self.function_label.configure(text="Función T: ⏳ No analizado", text_color="lightblue")
        self.current_result = None
        self.current_code = ""
        self.export_report_btn.configure(state="disabled")
        self.export_graph_btn.configure(state="disabled")

    def analyze_code(self):
        """Analiza el código ingresado"""
        code = self.code_text.get("1.0", "end-1c").strip()

        if not code:
            messagebox.showwarning("⚠️ Advertencia", "Por favor ingrese código para analizar")
            return

        try:
            # Mostrar estado de carga
            self.complexity_label.configure(text="Complejidad: 🔄 Analizando...", text_color="yellow")
            self.function_label.configure(text="Función T: 🔄 Procesando...", text_color="yellow")
            self.root.update()

            # Analizar el código
            self.current_result = self.analyzer.analyze_nested_loops(code)
            self.current_code = code

            # Actualizar interfaz con resultados
            self.complexity_label.configure(
                text=f"Complejidad: ✅ {self.current_result['complexity']}",
                text_color="lightgreen"
            )
            self.function_label.configure(
                text=f"Función T: {self.current_result['function'][:60]}{'...' if len(self.current_result['function']) > 60 else ''}",
                text_color="lightblue"
            )

            # Mostrar reporte completo en la interfaz
            self.results_text.delete("1.0", "end")
            formatted_report = self.generate_full_report()
            self.results_text.insert("1.0", formatted_report)

            # Habilitar botones de exportación
            self.export_report_btn.configure(state="normal")
            self.export_graph_btn.configure(state="normal")

            messagebox.showinfo("✅ Éxito",
                                f"Análisis completado correctamente!\n\n" +
                                f"Complejidad: {self.current_result['complexity']}\n" +
                                f"Bucles detectados: {self.current_result['loops_count']}")

        except Exception as e:
            self.complexity_label.configure(text="Complejidad: ❌ Error en análisis", text_color="red")
            self.function_label.configure(text="Función T: ❌ Error en análisis", text_color="red")
            messagebox.showerror("❌ Error", f"Error al analizar el código:\n\n{str(e)}")

    def open_comparison_window(self):
        """Abre la ventana de comparación de algoritmos"""
        comparison_window = ctk.CTkToplevel(self.root)
        comparison_window.title("⚖️ Comparador de Algoritmos")
        comparison_window.geometry("1400x800")
        comparison_window.resizable(True, True)  # Permitir maximizar
        comparison_window.grab_set()  # Hacer modal

        # Variables para almacenar los algoritmos
        algorithms_data = {}
        algorithm_frames = []
        algorithm_texts = []
        algorithm_results_labels = []
        algorithm_count = 1  # Empezar con 1 algoritmo

        # Título principal
        title_frame = ctk.CTkFrame(comparison_window)
        title_frame.pack(fill="x", padx=20, pady=(20, 10))

        title_label = ctk.CTkLabel(
            title_frame,
            text="⚖️ Comparador de Algoritmos",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=15)

        # Frame para controles de algoritmos (+/-)
        control_frame = ctk.CTkFrame(comparison_window)
        control_frame.pack(fill="x", padx=20, pady=(0, 10))

        # Label para mostrar cantidad actual
        count_label = ctk.CTkLabel(
            control_frame,
            text="Algoritmos a comparar: 1",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        count_label.pack(side="left", padx=(20, 20), pady=15)

        # Botón para agregar algoritmo
        add_btn = ctk.CTkButton(
            control_frame,
            text="➕ Agregar",
            command=lambda: add_algorithm(),
            width=120,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        add_btn.pack(side="left", padx=5, pady=15)

        # Botón para quitar algoritmo
        remove_btn = ctk.CTkButton(
            control_frame,
            text="➖ Quitar",
            command=lambda: remove_algorithm(),
            width=120,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="red",
            hover_color="darkred",
            state="disabled"  # Inicia deshabilitado porque solo hay 1
        )
        remove_btn.pack(side="left", padx=5, pady=15)

        # Botón principal: Analizar y Comparar (movido aquí)
        analyze_and_compare_btn = ctk.CTkButton(
            control_frame,
            text="🔍📊 Analizar y Comparar",
            command=lambda: analyze_all_and_compare(),
            width=220,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="blue",
            hover_color="darkblue"
        )
        analyze_and_compare_btn.pack(side="left", padx=(20, 10), pady=15)

        # Frame principal con scrollbar
        main_container = ctk.CTkFrame(comparison_window)
        main_container.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Crear scrollable frame
        scrollable_frame = ctk.CTkScrollableFrame(main_container, height=400)
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Función para crear un frame de algoritmo
        def create_algorithm_frame(index):
            # Frame contenedor del algoritmo
            algo_frame = ctk.CTkFrame(scrollable_frame)
            algo_frame.pack(fill="x", padx=10, pady=10)

            # Header del algoritmo
            header_frame = ctk.CTkFrame(algo_frame)
            header_frame.pack(fill="x", padx=15, pady=(15, 10))

            # Nombre del algoritmo (automático)
            name_label = ctk.CTkLabel(
                header_frame,
                text=f"🔢 Algoritmo {index + 1}",
                font=ctk.CTkFont(size=16, weight="bold")
            )
            name_label.pack(side="left", padx=15, pady=10)

            # Área de texto para el código
            code_frame = ctk.CTkFrame(algo_frame)
            code_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

            code_text = ctk.CTkTextbox(
                code_frame,
                height=120,
                font=ctk.CTkFont(family="Consolas", size=10),
                wrap="none"
            )
            code_text.pack(fill="both", expand=True, padx=10, pady=10)

            # Label para mostrar el resultado
            result_label = ctk.CTkLabel(
                algo_frame,
                text="⏳ No analizado",
                font=ctk.CTkFont(size=11),
                text_color="orange"
            )
            result_label.pack(padx=15, pady=(0, 15))

            return algo_frame, code_text, result_label

        # Función para agregar algoritmo
        def add_algorithm():
            nonlocal algorithm_count
            if algorithm_count < 5:  # Máximo 5 algoritmos
                frame, text, label = create_algorithm_frame(algorithm_count)
                algorithm_frames.append(frame)
                algorithm_texts.append(text)
                algorithm_results_labels.append(label)
                algorithm_count += 1

                count_label.configure(text=f"Algoritmos a comparar: {algorithm_count}")

                # Habilitar botón quitar si hay más de 1
                if algorithm_count > 1:
                    remove_btn.configure(state="normal")

                # Deshabilitar botón agregar si llegamos al máximo
                if algorithm_count >= 5:
                    add_btn.configure(state="disabled")

        # Función para quitar algoritmo
        def remove_algorithm():
            nonlocal algorithm_count
            if algorithm_count > 1:  # Mínimo 1 algoritmo
                # Remover el último algoritmo
                last_frame = algorithm_frames.pop()
                last_frame.destroy()
                algorithm_texts.pop()
                algorithm_results_labels.pop()
                algorithm_count -= 1

                count_label.configure(text=f"Algoritmos a comparar: {algorithm_count}")

                # Limpiar datos del algoritmo removido
                if algorithm_count in algorithms_data:
                    del algorithms_data[algorithm_count]

                # Deshabilitar botón quitar si solo queda 1
                if algorithm_count <= 1:
                    remove_btn.configure(state="disabled")

                # Habilitar botón agregar
                add_btn.configure(state="normal")

        # Crear el primer algoritmo
        frame, text, label = create_algorithm_frame(0)
        algorithm_frames.append(frame)
        algorithm_texts.append(text)
        algorithm_results_labels.append(label)

        # Frame para botones de acción
        action_frame = ctk.CTkFrame(comparison_window)
        action_frame.pack(fill="x", padx=20, pady=(0, 20))

        # Botón para cargar ejemplos
        load_examples_btn = ctk.CTkButton(
            action_frame,
            text="📋 Cargar Ejemplos",
            command=lambda: load_example_algorithms(),
            width=150,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="purple",
            hover_color="darkmagenta"
        )
        load_examples_btn.pack(side="left", padx=(20, 10), pady=15)

        # Botón para limpiar todo
        clear_all_btn = ctk.CTkButton(
            action_frame,
            text="🗑️ Limpiar",
            command=lambda: clear_all_algorithms(),
            width=120,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="gray",
            hover_color="darkgray"
        )
        clear_all_btn.pack(side="left", padx=10, pady=15)

        # Botón cerrar
        close_btn = ctk.CTkButton(
            action_frame,
            text="❌ Cerrar",
            command=comparison_window.destroy,
            width=100,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="red",
            hover_color="darkred"
        )
        close_btn.pack(side="right", padx=(10, 20), pady=15)

        def analyze_all_and_compare():
            """Analiza todos los algoritmos y genera la comparación"""
            algorithms_data.clear()

            # Analizar cada algoritmo
            valid_count = 0
            for i in range(algorithm_count):
                code = algorithm_texts[i].get("1.0", "end-1c").strip()

                # Verificar si no es el placeholder text
                placeholder = f"// Ingrese el código del Algoritmo {i + 1} aquí..."
                if code and code != placeholder:
                    try:
                        algorithm_results_labels[i].configure(text="🔄 Analizando...", text_color="yellow")
                        comparison_window.update()

                        result = self.analyzer.analyze_nested_loops(code)

                        algorithms_data[i] = {
                            'name': f"Algoritmo {i + 1}",
                            'code': code,
                            'result': result
                        }

                        algorithm_results_labels[i].configure(
                            text=f"✅ {result['complexity']}",
                            text_color="lightgreen"
                        )
                        valid_count += 1

                    except Exception as e:
                        algorithm_results_labels[i].configure(
                            text=f"❌ Error: {str(e)[:25]}...",
                            text_color="red"
                        )
                else:
                    algorithm_results_labels[i].configure(text="⚠️ Código vacío", text_color="orange")

            # Verificar que haya al menos 1 algoritmo válido para graficar
            if valid_count == 0:
                messagebox.showwarning("⚠️ Advertencia", "No hay algoritmos válidos para analizar")
                return

            # Generar la gráfica automáticamente
            generate_comparison_graph(algorithms_data)

        def load_example_algorithms():
            """Carga algoritmos de ejemplo"""
            examples = [
                '''for(int i = 0; i < n; i++) {
    if(array[i] == target) {
        return i;
    }
}''',
                '''while(left <= right) {
    int mid = (left + right) / 2;
    if(array[mid] == target) {
        return mid;
    }
    if(array[mid] < target) {
        left = mid + 1;
    } else {
        right = mid - 1;
    }
}''',
                '''for(int i = 0; i < n; i++) {
    for(int j = 0; j < n-1; j++) {
        if(array[j] > array[j+1]) {
            int temp = array[j];
            array[j] = array[j+1];
            array[j+1] = temp;
        }
    }
}''',
                '''for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
        for(int k = 0; k < n; k++) {
            if(array[i] + array[j] + array[k] == target) {
                return true;
            }
        }
    }
}''',
                '''for(int size = 1; size < n; size = size * 2) {
    for(int left = 0; left < n; left = left + size * 2) {
        int mid = left + size - 1;
        int right = left + size * 2 - 1;
        merge(array, left, mid, right);
    }
}'''
            ]

            # Ajustar la cantidad de algoritmos para los ejemplos
            target_count = min(5, len(examples))

            # Agregar algoritmos si es necesario
            while algorithm_count < target_count:
                add_algorithm()

            # Quitar algoritmos si es necesario
            while algorithm_count > target_count:
                remove_algorithm()

            # Cargar los ejemplos
            for i in range(min(algorithm_count, len(examples))):
                algorithm_texts[i].delete("1.0", "end")
                algorithm_texts[i].insert("1.0", examples[i])

        def clear_all_algorithms():
            """Limpia todos los algoritmos"""
            algorithms_data.clear()
            for i in range(algorithm_count):
                algorithm_texts[i].delete("1.0", "end")
                algorithm_results_labels[i].configure(text="⏳ No analizado", text_color="orange")

        def generate_comparison_graph(algos_data):
            """Genera la gráfica de comparación"""
            valid_algorithms = {k: v for k, v in algos_data.items() if v is not None}

            if len(valid_algorithms) == 0:
                messagebox.showwarning("⚠️ Advertencia", "No hay algoritmos válidos para comparar")
                return

            # Crear ventana de gráfica
            graph_window = ctk.CTkToplevel(comparison_window)
            graph_window.title("📊 Comparación de Algoritmos")
            graph_window.geometry("1400x800")
            graph_window.resizable(True, True)
            graph_window.grab_set()

            # Configurar estilo
            plt.style.use('dark_background')

            # Crear figura con una sola gráfica grande
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            fig.patch.set_facecolor('black')

            # Colores para cada algoritmo
            colors = ['cyan', 'yellow', 'lime', 'magenta', 'orange']
            line_styles = ['-', '--', '-.', ':', '-']

            n_values = np.logspace(1, 4, 100)  # De 10 a 10,000

            # Plotear todas las líneas de complejidad
            for i, (index, algo_data) in enumerate(valid_algorithms.items()):
                complexity = algo_data['result']['complexity']
                name = algo_data['name']
                color = colors[i % len(colors)]
                style = line_styles[i % len(line_styles)]

                y_values = self.get_complexity_values(n_values, complexity)
                ax.plot(n_values, y_values, color=color, linewidth=3,
                        linestyle=style, label=f'{name}: {complexity}', alpha=0.9)

            ax.set_xlabel('Tamaño de entrada (n)', fontsize=16, color='white', fontweight='bold')
            ax.set_ylabel('Tiempo de ejecución T(n)', fontsize=16, color='white', fontweight='bold')
            ax.set_title('Comparación de Complejidades Temporales', fontsize=18, fontweight='bold', color='cyan')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, color='gray')
            ax.legend(fontsize=12, loc='upper left', framealpha=0.8)

            # Mejorar el aspecto visual
            ax.tick_params(axis='both', which='major', labelsize=12, colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')

            plt.tight_layout()

            # Crear interfaz de la ventana
            title_frame = ctk.CTkFrame(graph_window)
            title_frame.pack(fill="x", padx=20, pady=(20, 10))

            title_label = ctk.CTkLabel(
                title_frame,
                text=f"📊 Comparación de {len(valid_algorithms)} Algoritmo{'s' if len(valid_algorithms) > 1 else ''}",
                font=ctk.CTkFont(size=20, weight="bold")
            )
            title_label.pack(pady=15)

            # Canvas para la gráfica
            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=(0, 10))

            # Frame para información resumida
            info_frame = ctk.CTkFrame(graph_window)
            info_frame.pack(fill="x", padx=20, pady=(0, 10))

            info_text = "Resumen: "
            for i, (index, algo_data) in enumerate(valid_algorithms.items()):
                if i > 0:
                    info_text += " | "
                info_text += f"{algo_data['name']}: {algo_data['result']['complexity']}"

            info_label = ctk.CTkLabel(
                info_frame,
                text=info_text,
                font=ctk.CTkFont(size=12),
                text_color="lightblue",
                wraplength=1200
            )
            info_label.pack(pady=10)

            # Botones
            button_frame = ctk.CTkFrame(graph_window)
            button_frame.pack(fill="x", padx=20, pady=(0, 20))

            save_comparison_btn = ctk.CTkButton(
                button_frame,
                text="💾 Guardar Comparación",
                command=lambda: self.save_comparison_graph(fig),
                width=180,
                height=35,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color="green",
                hover_color="darkgreen"
            )
            save_comparison_btn.pack(side="left", padx=(20, 10), pady=15)

            close_graph_btn = ctk.CTkButton(
                button_frame,
                text="❌ Cerrar",
                command=graph_window.destroy,
                width=120,
                height=35,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color="red",
                hover_color="darkred"
            )
            close_graph_btn.pack(side="right", padx=(10, 20), pady=15)

            # Centrar ventana de gráfica un poco más abajo
            graph_window.update_idletasks()
            x = (graph_window.winfo_screenwidth() // 2) - (1400 // 2)
            y = (graph_window.winfo_screenheight() // 2) - (400 // 2) + 50  # +50 para bajarla
            graph_window.geometry(f"1400x800+{x}+{y}")

        # Centrar ventana de comparación un poco más abajo
        comparison_window.update_idletasks()
        x = (comparison_window.winfo_screenwidth() // 2) - (1400 // 2)
        y = (comparison_window.winfo_screenheight() // 2) - (400 // 2) + 30  # +30 para bajarla
        comparison_window.geometry(f"1400x800+{x}+{y}")

    def get_complexity_values(self, n_values, complexity):
        """Obtiene los valores Y para graficar según la complejidad"""
        if "N^3" in complexity:
            return n_values ** 3
        elif "N^2" in complexity:
            return n_values ** 2
        elif "N × log^2 N" in complexity or "N log^2 N" in complexity:
            return n_values * (np.log2(n_values)) ** 2
        elif "N × log N" in complexity or "N log N" in complexity:
            return n_values * np.log2(n_values)
        elif "log^2 N" in complexity:
            return (np.log2(n_values)) ** 2
        elif "log N" in complexity:
            return np.log2(n_values)
        elif "N" in complexity:
            return n_values
        else:
            return np.ones_like(n_values)  # O(1)

    def save_comparison_graph(self, fig):
        """Guarda la gráfica de comparación"""
        file_path = filedialog.asksaveasfilename(
            title="Guardar comparación de algoritmos",
            defaultextension=".png",
            filetypes=[
                ("Archivos PNG", "*.png"),
                ("Archivos JPG", "*.jpg"),
                ("Archivos SVG", "*.svg"),
                ("Archivos PDF", "*.pdf")
            ],
            initialfile=f"comparacion_algoritmos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )

        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
                messagebox.showinfo("✅ Éxito", f"Comparación guardada exitosamente:\n{os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("❌ Error", f"Error al guardar comparación:\n{str(e)}")

    def generate_full_report(self):
        """Genera el reporte completo para mostrar en la interfaz"""
        report = []

        # Encabezado
        report.append("=" * 60)
        report.append("🔬 REPORTE COMPLETO DE ANÁLISIS DE COMPLEJIDAD TEMPORAL")
        report.append("=" * 60)
        report.append("")

        # Información general
        report.append("📋 INFORMACIÓN GENERAL")
        report.append("-" * 50)
        report.append(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append(f"Complejidad detectada: {self.current_result['complexity']}")
        report.append(f"Número de bucles anidados: {self.current_result['loops_count']}")
        report.append("")

        # Función de tiempo
        report.append("⚙️ FUNCIÓN DE TIEMPO FINAL")
        report.append("-" * 50)
        report.append(f"T(n) = {self.current_result['function']}")
        report.append("")

        # Análisis detallado
        report.append("📄 ANÁLISIS DETALLADO")
        report.append("-" * 50)
        report.append(self.current_result['analysis'])
        report.append("")

        # Código analizado
        report.append("💻 CÓDIGO ANALIZADO")
        report.append("-" * 50)
        code_lines = self.current_code.split('\n')
        for i, line in enumerate(code_lines, 1):
            if line.strip():
                report.append(f"{i:3d}: {line}")

        report.append("")
        report.append("=" * 60)
        report.append("📌 Fin del reporte")
        report.append("=" * 60)

        return "\n".join(report)

    def export_report(self):
        """Exporta el reporte completo a PDF"""
        if not self.current_result:
            messagebox.showwarning("⚠️ Advertencia", "No hay resultados para exportar")
            return

        file_path = filedialog.asksaveasfilename(
            title="Guardar reporte PDF",
            defaultextension=".pdf",
            filetypes=[("Archivos PDF", "*.pdf")],
            initialfile=f"reporte_complejidad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        if file_path:
            try:
                self.create_pdf_report(file_path)
                messagebox.showinfo("✅ Éxito", f"Reporte PDF guardado exitosamente:\n{os.path.basename(file_path)}")

                # Preguntar si quiere abrir el archivo
                if messagebox.askyesno("📄 Abrir archivo", "¿Desea abrir el reporte PDF generado?"):
                    try:
                        os.startfile(file_path)  # Windows
                    except:
                        # Para sistemas que no soporten startfile
                        import subprocess
                        subprocess.run(['xdg-open', file_path])  # Linux

            except Exception as e:
                messagebox.showerror("❌ Error", f"Error al generar PDF:\n{str(e)}")

    def create_pdf_report(self, file_path):
        """Crea el reporte PDF detallado con formato personalizado"""
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Estilo personalizado para título (Times New Roman 12pt, azul)
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName='Times-Bold',
            fontSize=12,
            spaceAfter=30,
            alignment=1,  # Centrado
            textColor='blue'
        )

        # Estilo para subtítulos (Times New Roman 12pt, verde, negrita)
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontName='Times-Bold',
            fontSize=12,
            spaceBefore=20,
            spaceAfter=10,
            textColor='green'
        )

        # Estilo para texto normal (Times New Roman 12pt)
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName='Times-Roman',
            fontSize=12
        )

        # Estilo para código (Courier)
        code_style = ParagraphStyle(
            'CustomCode',
            parent=styles['Code'],
            fontName='Courier',
            fontSize=10
        )

        # Título principal
        story.append(Paragraph("Reporte de Análisis de Complejidad Temporal", title_style))
        story.append(Spacer(1, 20))

        # Información general
        story.append(Paragraph("Información General", subtitle_style))
        story.append(
            Paragraph(f"<b>Fecha de análisis:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", normal_style))
        story.append(Paragraph(f"<b>Complejidad detectada:</b> {self.current_result['complexity']}", normal_style))
        story.append(
            Paragraph(f"<b>Número de bucles anidados:</b> {self.current_result['loops_count']}", normal_style))
        story.append(Spacer(1, 20))

        # ALGORITMO ANALIZADO
        story.append(Paragraph("ALGORITMO ANALIZADO", subtitle_style))
        story.append(Paragraph("─" * 30, normal_style))

        # Agregar código línea por línea con numeración
        code_lines = self.current_code.split('\n')
        for i, line in enumerate(code_lines, 1):
            if line.strip():  # Solo líneas no vacías
                story.append(Paragraph(f"<font name='Courier' size='10'>{i:2d}: {line}</font>", normal_style))
        story.append(Paragraph("─" * 30, normal_style))
        story.append(Spacer(1, 20))

        # Función de tiempo
        story.append(Paragraph("Función de Tiempo", subtitle_style))
        story.append(
            Paragraph(f"<font name='Courier' size='10'>{self.current_result['function']}</font>", normal_style))
        story.append(Spacer(1, 20))

        # Pasos del análisis
        story.append(Paragraph("Pasos del Análisis", subtitle_style))
        for i, step in enumerate(self.current_result['steps'], 1):
            story.append(Paragraph(f"<b>{i}.</b> {step}", normal_style))
        story.append(Spacer(1, 20))

        # Análisis detallado
        story.append(Paragraph("Análisis Detallado", subtitle_style))
        analysis_lines = self.current_result['analysis'].split('\n')
        for line in analysis_lines:
            if line.strip():
                if line.startswith('===') or line.startswith('---'):
                    story.append(Paragraph(f"<b>{line}</b>", normal_style))
                else:
                    story.append(Paragraph(f"<font name='Courier' size='9'>{line}</font>", normal_style))

        # Pie de página con información adicional
        story.append(Spacer(1, 30))
        story.append(Paragraph("Notas", subtitle_style))
        story.append(Paragraph("• Este reporte fue generado automáticamente por el Analizador de Complejidad Temporal",
                               normal_style))
        story.append(Paragraph("• La complejidad calculada es asintótica (comportamiento para valores grandes de n)",
                               normal_style))
        story.append(Paragraph("• Los resultados pueden variar según la implementación específica del hardware",
                               normal_style))
        story.append(
            Paragraph("• El cálculo se realiza de ADENTRO hacia AFUERA (bucle más interno primero)", normal_style))

        doc.build(story)

    def show_graph(self):
        """Muestra la gráfica de complejidad en una nueva ventana"""
        if not self.current_result:
            messagebox.showwarning("⚠️ Advertencia", "No hay resultados para graficar")
            return

        try:
            self.create_and_show_graph()

        except Exception as e:
            messagebox.showerror("❌ Error", f"Error al generar gráfica:\n{str(e)}")

    def create_and_show_graph(self):
        """Crea y muestra la gráfica de complejidad temporal en una ventana nueva"""
        # Crear nueva ventana para la gráfica
        graph_window = ctk.CTkToplevel(self.root)
        graph_window.title("📊 Gráfica de Complejidad Temporal")
        graph_window.geometry("1200x700")
        graph_window.grab_set()  # Hacer la ventana modal

        # Configurar el estilo para fondo oscuro
        plt.style.use('dark_background')

        # Crear la figura con subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('black')

        # Generar datos para diferentes tamaños de entrada
        n_values = np.logspace(1, 4, 100)  # De 10 a 10,000

        # Extraer el tipo de complejidad
        complexity = self.current_result['complexity']

        # Calcular valores según la complejidad detectada
        complexity_functions = self.get_complexity_functions(n_values, complexity)

        # GRÁFICA 1: Complejidad principal
        for func_name, y_values, color, style in complexity_functions:
            ax1.plot(n_values, y_values, color=color, linewidth=3,
                     linestyle=style, label=func_name, alpha=0.8)

        ax1.set_xlabel('Tamaño de entrada (n)', fontsize=14, color='white', fontweight='bold')
        ax1.set_ylabel('Tiempo de ejecución T(n)', fontsize=14, color='white', fontweight='bold')
        ax1.set_title(f'Complejidad Temporal: {complexity}', fontsize=16, fontweight='bold', color='cyan')

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.legend(fontsize=11, loc='upper left')

        # Agregar información de la función
        info_text = f'Función: {self.current_result["function"][:60]}...' if len(
            self.current_result["function"]) > 60 else f'Función: {self.current_result["function"]}'
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                 color='white', wrap=True)

        # GRÁFICA 2: Comparación con otras complejidades comunes
        self.create_comparison_graph(ax2, n_values, complexity)

        # Ajustar layout
        plt.tight_layout()

        # Crear el canvas para mostrar en la ventana de tkinter
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()

        # Frame para el título de la ventana
        title_frame = ctk.CTkFrame(graph_window)
        title_frame.pack(fill="x", padx=20, pady=(20, 10))

        title_label = ctk.CTkLabel(
            title_frame,
            text=f"📊 Análisis Gráfico - {complexity}",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=15)

        # Frame para información adicional
        info_frame = ctk.CTkFrame(graph_window)
        info_frame.pack(fill="x", padx=20, pady=(0, 10))

        info_label = ctk.CTkLabel(
            info_frame,
            text=f"Función de tiempo: {self.current_result['function'][:80]}{'...' if len(self.current_result['function']) > 80 else ''}",
            font=ctk.CTkFont(size=12),
            text_color="lightblue"
        )
        info_label.pack(pady=10)

        # Empaquetar el canvas
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Frame para botones
        button_frame = ctk.CTkFrame(graph_window)
        button_frame.pack(fill="x", padx=20, pady=(0, 20))

        # Botón para guardar gráfica
        save_btn = ctk.CTkButton(
            button_frame,
            text="💾 Guardar Gráfica",
            command=lambda: self.save_current_graph(fig),
            width=150,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        save_btn.pack(side="left", padx=(20, 10), pady=15)

        # Botón para cerrar
        close_btn = ctk.CTkButton(
            button_frame,
            text="❌ Cerrar",
            command=graph_window.destroy,
            width=120,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="red",
            hover_color="darkred"
        )
        close_btn.pack(side="right", padx=(10, 20), pady=15)

        # Centrar la ventana
        graph_window.update_idletasks()
        x = (graph_window.winfo_screenwidth() // 2) - (1200 // 2)
        y = (graph_window.winfo_screenheight() // 2) - (700 // 2)
        graph_window.geometry(f"1200x700+{x}+{y}")

    def save_current_graph(self, fig):
        """Guarda la gráfica actual"""
        file_path = filedialog.asksaveasfilename(
            title="Guardar gráfica de complejidad",
            defaultextension=".png",
            filetypes=[
                ("Archivos PNG", "*.png"),
                ("Archivos JPG", "*.jpg"),
                ("Archivos SVG", "*.svg"),
                ("Archivos PDF", "*.pdf")
            ],
            initialfile=f"grafica_complejidad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )

        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
                messagebox.showinfo("✅ Éxito", f"Gráfica guardada exitosamente:\n{os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("❌ Error", f"Error al guardar gráfica:\n{str(e)}")

    def get_complexity_functions(self, n_values, complexity):
        """Obtiene las funciones de complejidad para graficar"""
        functions = []

        # Función principal detectada
        if "log^2 N" in complexity and "N^2" in complexity:
            y_main = n_values ** 2 * (np.log2(n_values)) ** 2
            functions.append((f'Tu algoritmo: {complexity}', y_main, 'cyan', '-'))
        elif "N × log^2 N" in complexity or "N log^2 N" in complexity:
            y_main = n_values * (np.log2(n_values)) ** 2
            functions.append((f'Tu algoritmo: {complexity}', y_main, 'cyan', '-'))
        elif "N × log N" in complexity or "N log N" in complexity:
            y_main = n_values * np.log2(n_values)
            functions.append((f'Tu algoritmo: {complexity}', y_main, 'cyan', '-'))
        elif "N^3" in complexity:
            y_main = n_values ** 3
            functions.append((f'Tu algoritmo: {complexity}', y_main, 'cyan', '-'))
        elif "N^2" in complexity:
            y_main = n_values ** 2
            functions.append((f'Tu algoritmo: {complexity}', y_main, 'cyan', '-'))
        elif "log N" in complexity:
            y_main = np.log2(n_values)
            functions.append((f'Tu algoritmo: {complexity}', y_main, 'cyan', '-'))
        else:
            y_main = n_values
            functions.append((f'Tu algoritmo: {complexity}', y_main, 'cyan', '-'))

        return functions

    def create_comparison_graph(self, ax, n_values, main_complexity):
        """Crea gráfica de comparación con complejidades estándar"""
        # Complejidades comunes para comparación
        complexities = {
            'O(1)': (np.ones_like(n_values), 'yellow', '--'),
            'O(log N)': (np.log2(n_values), 'green', '--'),
            'O(N)': (n_values, 'blue', '--'),
            'O(N log N)': (n_values * np.log2(n_values), 'orange', '--'),
            'O(N²)': (n_values ** 2, 'red', '--'),
            'O(N³)': (n_values ** 3, 'purple', '--')
        }

        # Plotear complejidades comunes
        for comp_name, (y_vals, color, style) in complexities.items():
            ax.plot(n_values, y_vals, color=color, linewidth=2,
                    linestyle=style, label=comp_name, alpha=0.7)

        # Plotear la complejidad detectada con línea sólida y más gruesa
        if "N × log^2 N" in main_complexity:
            y_main = n_values * (np.log2(n_values)) ** 2
            ax.plot(n_values, y_main, color='cyan', linewidth=4,
                    label=f'Tu algoritmo: {main_complexity}', alpha=1.0)
        elif "N × log N" in main_complexity:
            y_main = n_values * np.log2(n_values)
            ax.plot(n_values, y_main, color='cyan', linewidth=4,
                    label=f'Tu algoritmo: {main_complexity}', alpha=1.0)
        elif "N^2" in main_complexity:
            y_main = n_values ** 2
            ax.plot(n_values, y_main, color='cyan', linewidth=4,
                    label=f'Tu algoritmo: {main_complexity}', alpha=1.0)

        ax.set_xlabel('Tamaño de entrada (n)', fontsize=14, color='white', fontweight='bold')
        ax.set_ylabel('Tiempo relativo', fontsize=14, color='white', fontweight='bold')
        ax.set_title('Comparación con Complejidades Estándar', fontsize=16, fontweight='bold', color='lightgreen')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, color='gray')
        ax.legend(fontsize=10, loc='upper left')

        # Agregar nota explicativa
        note = "Las líneas punteadas muestran complejidades estándar.\nLa línea sólida cyan es tu algoritmo."
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                color='white')

    def run(self):
        """Ejecuta la aplicación"""
        # Configurar el icono de la ventana (opcional)
        try:
            self.root.iconbitmap("icon.ico")  # Si tienes un icono
        except:
            pass

        self.root.mainloop()


def create_gui():
    """Función de conveniencia para crear la interfaz gráfica"""
    return ComplexityAnalyzerGUI()