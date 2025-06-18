# gui.py - Interfaz Gráfica para el Analizador de Complejidad Temporal
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

        # Frame para botones de entrada
        input_buttons_frame = ctk.CTkFrame(left_panel)
        input_buttons_frame.pack(fill="x", padx=20, pady=(0, 15))

        self.load_file_btn = ctk.CTkButton(
            input_buttons_frame,
            text="📁 Cargar Archivo",
            command=self.load_file,
            width=140,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.load_file_btn.pack(side="left", padx=(15, 8), pady=12)

        self.clear_btn = ctk.CTkButton(
            input_buttons_frame,
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
            input_buttons_frame,
            text="🔍 Analizar",
            command=self.analyze_code,
            width=140,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.analyze_btn.pack(side="left", padx=8, pady=12)

        # Botones de exportación junto al botón analizar
        self.export_report_btn = ctk.CTkButton(
            input_buttons_frame,
            text="📋 PDF",
            command=self.export_report,
            width=100,
            height=35,
            state="disabled",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="blue",
            hover_color="darkblue"
        )
        self.export_report_btn.pack(side="left", padx=5, pady=12)

        self.export_graph_btn = ctk.CTkButton(
            input_buttons_frame,
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

    def generate_full_report(self):
        """Genera el reporte completo para mostrar en la interfaz"""
        report = []

        # Encabezado
        report.append("=" * 80)
        report.append("🔬 REPORTE COMPLETO DE ANÁLISIS DE COMPLEJIDAD TEMPORAL")
        report.append("=" * 80)
        report.append("")

        # Información general
        report.append("📋 INFORMACIÓN GENERAL")
        report.append("-" * 50)
        report.append(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append(f"Complejidad detectada: {self.current_result['complexity']}")
        report.append(f"Número de bucles anidados: {self.current_result['loops_count']}")
        report.append("")

        # Función de tiempo
        report.append("⚙️ FUNCIÓN DE TIEMPO")
        report.append("-" * 50)
        report.append(f"T(n) = {self.current_result['function']}")
        report.append("")

        # Pasos del análisis
        report.append("📊 PASOS DEL ANÁLISIS")
        report.append("-" * 50)
        for i, step in enumerate(self.current_result['steps'], 1):
            report.append(f"{i}. {step}")
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
        report.append("=" * 80)
        report.append("📌 Fin del reporte")
        report.append("=" * 80)

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
        """Crea el reporte PDF detallado"""
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Estilo personalizado para título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=1,  # Centrado
            textColor='darkblue'
        )

        # Estilo para subtítulos
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor='darkgreen'
        )

        # Título principal
        story.append(Paragraph("🔬 Reporte de Análisis de Complejidad Temporal", title_style))
        story.append(Spacer(1, 20))

        # Información general
        story.append(Paragraph("📋 Información General", subtitle_style))
        story.append(
            Paragraph(f"<b>Fecha de análisis:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Complejidad detectada:</b> {self.current_result['complexity']}", styles['Normal']))
        story.append(
            Paragraph(f"<b>Número de bucles anidados:</b> {self.current_result['loops_count']}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Función de tiempo
        story.append(Paragraph("⚙️ Función de Tiempo", subtitle_style))
        story.append(
            Paragraph(f"<font name='Courier' size='10'>{self.current_result['function']}</font>", styles['Normal']))
        story.append(Spacer(1, 20))

        # Pasos del análisis
        story.append(Paragraph("📊 Pasos del Análisis", subtitle_style))
        for i, step in enumerate(self.current_result['steps'], 1):
            story.append(Paragraph(f"<b>{i}.</b> {step}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Análisis detallado
        story.append(Paragraph("📄 Análisis Detallado", subtitle_style))
        analysis_lines = self.current_result['analysis'].split('\n')
        for line in analysis_lines:
            if line.strip():
                if line.startswith('===') or line.startswith('---'):
                    story.append(Paragraph(f"<b>{line}</b>", styles['Normal']))
                else:
                    story.append(Paragraph(f"<font name='Courier' size='9'>{line}</font>", styles['Normal']))

        # Nueva página para el código
        story.append(PageBreak())
        story.append(Paragraph("💻 Código Analizado", subtitle_style))

        # Agregar código línea por línea
        code_lines = self.current_code.split('\n')
        for i, line in enumerate(code_lines, 1):
            if line.strip():  # Solo líneas no vacías
                story.append(Paragraph(f"<font name='Courier' size='8'>{i:3d}: {line}</font>", styles['Code']))

        # Pie de página con información adicional
        story.append(Spacer(1, 30))
        story.append(Paragraph("📌 Notas", subtitle_style))
        story.append(Paragraph("• Este reporte fue generado automáticamente por el Analizador de Complejidad Temporal",
                               styles['Normal']))
        story.append(Paragraph("• La complejidad calculada es asintótica (comportamiento para valores grandes de n)",
                               styles['Normal']))
        story.append(Paragraph("• Los resultados pueden variar según la implementación específica del hardware",
                               styles['Normal']))
        story.append(
            Paragraph("• El cálculo se realiza de ADENTRO hacia AFUERA (bucle más interno primero)", styles['Normal']))

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

        # Mensaje de bienvenida
        messagebox.showinfo("🎉 Bienvenido",
                            "¡Bienvenido al Analizador de Complejidad Temporal!\n\n" +
                            "Características:\n" +
                            "• Análisis automático de bucles FOR anidados\n" +
                            "• Soporte para declaraciones IF y RETURN\n" +
                            "• Cálculo correcto de ADENTRO hacia AFUERA\n" +
                            "• Reporte completo visible en interfaz\n" +
                            "• Generación de reportes PDF\n" +
                            "• Gráficas de complejidad comparativas\n" +
                            "• Interfaz intuitiva y moderna\n\n" +
                            "¡Comienza analizando el código de ejemplo!")

        self.root.mainloop()


def create_gui():
    """Función de conveniencia para crear la interfaz gráfica"""
    return ComplexityAnalyzerGUI()
