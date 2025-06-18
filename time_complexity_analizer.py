# time_complexity_analyzer.py - Analizador de Complejidad Temporal
import re
import math
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class LoopInfo:
    """Información sobre un bucle detectado en el código"""

    def __init__(self, loop_type: str, var: str, start: str, end: str, step: str, level: int):
        self.loop_type = loop_type
        self.var = var
        self.start = start
        self.end = end
        self.step = step
        self.level = level
        self.operations_count = 0
        self.inner_loops = []
        self.time_complexity = "1"
        self.increment_expression = ""
        self.complexity_type = "constant"

    def __str__(self):
        return f"Loop({self.loop_type}, {self.var}: {self.start} to {self.end} step {self.step}, level={self.level})"


class TimeComplexityAnalyzer:
    """Analizador principal de complejidad temporal para código C++"""

    def __init__(self):
        self.loops_stack = []
        self.current_level = 0
        self.operations_outside_loops = 0

    def analyze_for_increment_operations(self, increment_expr: str) -> int:
        """
        Analiza las operaciones en la expresión de incremento del FOR

        Args:
            increment_expr: Expresión de incremento (ej: "i++", "i+=2", "i=i*2")

        Returns:
            Número de operaciones detectadas
        """
        if not increment_expr.strip():
            return 0

        if '=' in increment_expr:
            right_side = increment_expr.split('=', 1)[1].strip()
            arith_ops = len(re.findall(r'[+\-*/]', right_side))
            total_ops = arith_ops + 1  # +1 por la asignación
            return total_ops
        return 0

    def count_if_statements(self, block: str) -> int:
        """
        Cuenta las declaraciones IF en un bloque de código y sus operaciones internas

        Args:
            block: Bloque de código a analizar

        Returns:
            Número total de operaciones en declaraciones IF
        """
        if not block.strip():
            return 0

        if_pattern = r'if\s*\(([^)]*(?:==|!=|<=|>=|<|>)[^)]*)\)'
        matches = re.findall(if_pattern, block, re.IGNORECASE)
        total_if_operations = 0

        if len(matches) > 0:
            for i, condition in enumerate(matches, 1):
                comparison_ops = 1  # Una comparación por IF
                arithmetic_ops = len(re.findall(r'[+\-*/]', condition))
                if_ops = comparison_ops + arithmetic_ops
                total_if_operations += if_ops

        return total_if_operations

    def parse_for_loop(self, for_statement: str) -> Optional[LoopInfo]:
        """
        Parsea una declaración for y extrae información del bucle

        Args:
            for_statement: Declaración for completa

        Returns:
            Objeto LoopInfo con la información del bucle o None si no se puede parsear
        """
        pattern = r'for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*([^;]+);\s*\1\s*([<>=!]+)\s*([^;]+);\s*(.+)\)'

        match = re.search(pattern, for_statement, re.IGNORECASE)
        if match:
            var = match.group(1)
            start = match.group(2).strip()
            operator = match.group(3).strip()
            end = match.group(4).strip()
            step_expr = match.group(5).strip()

            step = self._extract_step_info(step_expr, var)
            loop = LoopInfo('for', var, start, end, step, self.current_level)
            loop.increment_expression = step_expr
            loop.complexity_type = self._determine_complexity_type(step)

            return loop
        return None

    def _extract_step_info(self, step_expr: str, var: str) -> str:
        """
        Extrae información del paso del bucle

        Args:
            step_expr: Expresión de incremento/decremento
            var: Variable del bucle

        Returns:
            Información del paso simplificada
        """
        step_expr = step_expr.replace(' ', '')

        if f'{var}++' in step_expr or f'++{var}' in step_expr:
            return '+1'
        elif f'{var}--' in step_expr or f'--{var}' in step_expr:
            return '-1'
        elif f'{var}*' in step_expr:
            pattern = rf'{var}\*(\d+)'
            match = re.search(pattern, step_expr)
            if match:
                return f'*{match.group(1)}'
        elif f'{var}/' in step_expr:
            pattern = rf'{var}/(\d+)'
            match = re.search(pattern, step_expr)
            if match:
                return f'/{match.group(1)}'
        elif f'{var}+' in step_expr:
            pattern = rf'{var}\+(.+)'
            match = re.search(pattern, step_expr)
            if match:
                expr = match.group(1)
                try:
                    result = eval(expr.replace(var, '0'))
                    return f'+{result}'
                except:
                    return f'+{expr}'

        return step_expr

    def _determine_complexity_type(self, step: str) -> str:
        """
        Determina el tipo de complejidad basado en el paso del bucle

        Args:
            step: Información del paso del bucle

        Returns:
            Tipo de complejidad: "linear", "logarithmic", o "constant"
        """
        if step.startswith('+') or step.startswith('-'):
            return "linear"
        elif step.startswith('*') or step.startswith('/'):
            return "logarithmic"
        else:
            return "constant"

    def calculate_loop_cycles(self, loop: LoopInfo) -> str:
        """
        Calcula el número de ciclos de un bucle

        Args:
            loop: Información del bucle

        Returns:
            Expresión matemática del número de ciclos
        """
        start = loop.start
        end = loop.end
        step = loop.step

        if step == '+1':
            return f"({end} - {start} + 1)"
        elif step == '-1':
            return f"({start} - {end} + 1)"
        elif step.startswith('*'):
            multiplier = step[1:]
            return f"(log_{multiplier}({end}/{start}) + c)"
        elif step.startswith('/'):
            divisor = step[1:]
            return f"(log_{divisor}({start}/{end}) + c)"
        elif step.startswith('+'):
            increment = step[1:]
            return f"({end} - {start})/{increment}"
        else:
            return f"f({start}, {end}, {step})"

    def count_operations_in_block(self, block: str) -> int:
        """
        Cuenta las operaciones en un bloque de código, incluyendo IFs, returns y asignaciones

        Args:
            block: Bloque de código a analizar

        Returns:
            Número total de operaciones
        """
        if not block.strip():
            return 0

        operations = 0

        # Contar IFs con operaciones internas
        if_operations = self.count_if_statements(block)
        operations += if_operations

        # Contar returns
        return_count = len(re.findall(r'\breturn\b', block, re.IGNORECASE))
        operations += return_count

        # Contar asignaciones
        statements = [stmt.strip() for stmt in block.split(';') if stmt.strip()]

        for stmt in statements:
            if stmt.strip().startswith('if') or stmt.strip().startswith('return'):
                continue

            if '=' in stmt and not any(op in stmt for op in ['==', '!=', '<=', '>=']):
                parts = stmt.split('=', 1)
                if len(parts) == 2:
                    left_side = parts[0].strip()
                    right_side = parts[1].strip()

                    left_arith_ops = len(re.findall(r'[+\-*/]', left_side))
                    right_arith_ops = len(re.findall(r'[+\-*/]', right_side))

                    stmt_operations = 1 + left_arith_ops + right_arith_ops
                    operations += stmt_operations

        return max(operations, 1)

    def analyze_nested_loops(self, code: str) -> Dict:
        """
        Método principal: Analiza bucles anidados y calcula la complejidad temporal

        Args:
            code: Código fuente a analizar

        Returns:
            Diccionario con el análisis completo de complejidad
        """
        loops_info = self._identify_all_loops(code)

        for loop in loops_info:
            loop_content = self._extract_specific_loop_content(code, loop)
            base_operations = self.count_operations_in_block(loop_content)

            for_base = 1
            increment_ops = self.analyze_for_increment_operations(loop.increment_expression)
            total_for_ops = for_base + increment_ops

            loop.operations_count = base_operations + total_for_ops

        return self._calculate_time_complexity(loops_info)

    def _identify_all_loops(self, code: str) -> List[LoopInfo]:
        """
        Identifica todos los bucles y sus niveles de anidamiento

        Args:
            code: Código fuente

        Returns:
            Lista de objetos LoopInfo
        """
        lines = code.split('\n')
        loops_info = []
        current_level = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            if line.startswith('for'):
                loop = self.parse_for_loop(line)
                if loop:
                    loop.level = current_level
                    loops_info.append(loop)
                    current_level += 1

            elif '}' in line and current_level > 0:
                current_level -= 1

        return loops_info

    def _extract_specific_loop_content(self, code: str, target_loop: LoopInfo) -> str:
        """
        Extrae solo las operaciones directas de un bucle específico

        Args:
            code: Código fuente completo
            target_loop: Bucle objetivo

        Returns:
            Contenido del bucle específico
        """
        lines = code.split('\n')
        content = []
        inside_target = False
        current_level = 0
        target_found = False
        skip_inner_loop = False
        inner_loop_level = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            if line.startswith('for') and not target_found:
                current_loop = self.parse_for_loop(line)
                if (current_loop and current_loop.var == target_loop.var and
                        current_level == target_loop.level):
                    inside_target = True
                    target_found = True
                current_level += 1
                continue

            elif line.startswith('for') and inside_target:
                inner_loop = self.parse_for_loop(line)
                if inner_loop:
                    skip_inner_loop = True
                    inner_loop_level = current_level
                current_level += 1
                continue

            elif '}' in line:
                current_level -= 1
                if skip_inner_loop and current_level <= inner_loop_level:
                    skip_inner_loop = False
                elif inside_target and current_level <= target_loop.level:
                    break
                continue

            if inside_target and not skip_inner_loop:
                if (line.startswith('if') or
                        line.startswith('return') or
                        ('=' in line and not any(
                            op in line for op in ['==', '!=', '<=', '>=', '<', '>']) and not line.startswith('for')) or
                        line.endswith(';')):
                    content.append(line)

        result = ' ; '.join(content)
        return result

    def _calculate_time_complexity(self, loops_info: List[LoopInfo]) -> Dict:
        """
        Calcula la complejidad temporal total - de adentro hacia afuera

        Args:
            loops_info: Lista de bucles detectados

        Returns:
            Diccionario con el análisis completo
        """
        if not loops_info:
            return {"complexity": "O(1)", "function": "T = 1", "analysis": "Sin bucles detectados"}

        # Ordenar bucles por nivel (más interno primero para el cálculo correcto)
        loops_info.sort(key=lambda x: x.level, reverse=True)

        time_expressions = []
        analysis_steps = []

        total_complexity = self._calculate_nested_complexity(loops_info)
        tiempo_acumulado = None

        for i, loop in enumerate(loops_info):
            cycles = self.calculate_loop_cycles(loop)
            operations = loop.operations_count

            if i == 0:  # Bucle más interno
                time_expr = f"T_{loop.level + 1} = {cycles} * {operations} + 2"
                tiempo_acumulado = f"{cycles} * {operations} + 2"
            else:
                # Bucles externos incluyen el tiempo de bucles internos ya calculados
                time_expr = f"T_{loop.level + 1} = {cycles} * ({operations} + ({tiempo_acumulado})) + 2"
                tiempo_acumulado = f"{cycles} * ({operations} + ({tiempo_acumulado})) + 2"

            time_expressions.append(time_expr)
            analysis_steps.append(f"Nivel {loop.level}: {loop} -> {time_expr}")

        return {
            "complexity": total_complexity,
            "function": time_expressions[-1] if time_expressions else "T = 1",
            "steps": analysis_steps,
            "loops_count": len(loops_info),
            "analysis": self._generate_detailed_analysis(loops_info, time_expressions, total_complexity,
                                                         tiempo_acumulado)
        }

    def _calculate_nested_complexity(self, loops_info: List[LoopInfo]) -> str:
        """
        Calcula la complejidad de bucles anidados basándose en su estructura

        Args:
            loops_info: Lista de bucles

        Returns:
            Notación Big O de la complejidad
        """
        if not loops_info:
            return "O(1)"

        linear_count = 0
        log_count = 0

        for loop in loops_info:
            if loop.complexity_type == "linear":
                linear_count += 1
            elif loop.complexity_type == "logarithmic":
                log_count += 1

        complexity_parts = []

        if linear_count > 0:
            if linear_count == 1:
                complexity_parts.append("N")
            else:
                complexity_parts.append(f"N^{linear_count}")

        if log_count > 0:
            if log_count == 1:
                complexity_parts.append("log N")
            else:
                complexity_parts.append(f"log^{log_count} N")

        if not complexity_parts:
            return "O(1)"

        final_complexity = " × ".join(complexity_parts)
        return f"O({final_complexity})"

    def _generate_detailed_analysis(self, loops_info: List[LoopInfo], expressions: List[str],
                                    final_complexity: str, tiempo_final: str) -> str:
        """
        Genera análisis detallado con complejidad corregida

        Args:
            loops_info: Lista de bucles
            expressions: Expresiones de tiempo
            final_complexity: Complejidad final
            tiempo_final: Tiempo final calculado

        Returns:
            Análisis detallado en formato string
        """
        analysis = []
        analysis.append("=== ANÁLISIS DETALLADO ===")
        analysis.append("Fórmula: T = (numCiclos + c) × Tfor + 2")
        analysis.append("Nota: Operaciones del FOR = 1 (base) + operaciones del incremento")
        analysis.append("Nota: Cada IF cuenta como: 1 comparación + operaciones aritméticas")
        analysis.append("Nota: Cada RETURN cuenta como T = 1 operación")
        analysis.append("Nota: Cada asignación cuenta como: 1 asignación + operaciones aritméticas")
        analysis.append("IMPORTANTE: El cálculo se hace de ADENTRO hacia AFUERA")

        time_values = []

        for i, loop in enumerate(loops_info):
            for_base = 1
            increment_ops = self.analyze_for_increment_operations(loop.increment_expression)
            total_for_ops = for_base + increment_ops
            base_ops = loop.operations_count - total_for_ops

            analysis.append(f"\nBucle {len(loops_info) - i} (Nivel {loop.level}):")
            analysis.append(f"  Variable: {loop.var}")
            analysis.append(f"  Rango: {loop.start} hasta {loop.end}")
            analysis.append(f"  Paso: {loop.step}")
            analysis.append(f"  Tipo de complejidad: {loop.complexity_type}")
            analysis.append(f"  Operaciones del código: {base_ops}")
            analysis.append(f"  Operaciones del FOR: +{total_for_ops}")
            analysis.append(f"  Total operaciones propias: {loop.operations_count}")
            analysis.append(f"  Ciclos: {self.calculate_loop_cycles(loop)}")

            if i == 0:  # Bucle más interno
                cycles = self.calculate_loop_cycles(loop)
                ops = loop.operations_count
                result = f"T₍{loop.level + 1}₎ = {cycles} × {ops} + 2"
                analysis.append(f"  {result}")
                time_values.append(f"{cycles} × {ops} + 2")
            else:
                cycles = self.calculate_loop_cycles(loop)
                ops = loop.operations_count
                inner_value = time_values[i - 1]
                result = f"T₍{loop.level + 1}₎ = {cycles} × ({ops} + ({inner_value})) + 2"
                analysis.append(f"  {result}")
                time_values.append(f"{cycles} × ({ops} + ({inner_value})) + 2")

        if len(time_values) > 0:
            analysis.append(f"\nFunción tiempo final: T₍{loops_info[0].level + 1}₎ = {tiempo_final}")
            analysis.append(f"Complejidad asintótica: {final_complexity}")

            linear_loops = sum(1 for loop in loops_info if loop.complexity_type == "linear")
            log_loops = sum(1 for loop in loops_info if loop.complexity_type == "logarithmic")

            analysis.append(f"\nCálculo de complejidad:")
            analysis.append(f"  - Bucles lineales: {linear_loops}")
            analysis.append(f"  - Bucles logarítmicos: {log_loops}")

            if linear_loops > 0 and log_loops > 0:
                analysis.append(f"  - Combinación: N^{linear_loops} × log^{log_loops} N = {final_complexity}")
            elif linear_loops > 0:
                analysis.append(f"  - Solo lineales: N^{linear_loops} = {final_complexity}")
            elif log_loops > 0:
                analysis.append(f"  - Solo logarítmicos: log^{log_loops} N = {final_complexity}")

        return "\n".join(analysis)


# Función de utilidad para usar el analizador independientemente
def analyze_code_complexity(code: str) -> Dict:
    """
    Función de conveniencia para analizar código

    Args:
        code: Código fuente a analizar

    Returns:
        Análisis de complejidad
    """
    analyzer = TimeComplexityAnalyzer()
    return analyzer.analyze_nested_loops(code)
