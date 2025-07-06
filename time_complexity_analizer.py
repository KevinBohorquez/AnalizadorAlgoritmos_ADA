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
        self.condition = ""  # Para while loops

    def __str__(self):
        return f"Loop({self.loop_type}, {self.var}: {self.start} to {self.end} step {self.step}, level={self.level})"


class TimeComplexityAnalyzer:
    """Analizador principal de complejidad temporal para código C++"""

    def __init__(self):
        self.loops_stack = []
        self.current_level = 0
        self.operations_outside_loops = 0

    def analyze_for_increment_operations(self, increment_expr: str) -> int:
        """Analiza las operaciones en la expresión de incremento del FOR"""
        if not increment_expr.strip():
            return 0

        if '=' in increment_expr:
            right_side = increment_expr.split('=', 1)[1].strip()
            arith_ops = len(re.findall(r'[+\-*/]', right_side))
            total_ops = arith_ops + 1  # +1 por la asignación
            return total_ops
        return 0

    def analyze_while_increment_operations(self, block_content: str, var: str) -> Tuple[int, str]:
        """Analiza las operaciones de incremento en el cuerpo de un while"""
        if not block_content.strip():
            return 0, "+1"

        # Limpiar el contenido del bloque
        content = block_content.replace('\n', ' ').replace('\t', ' ')

        step = "+1"
        operations = 0

        # Patrón para i = i * expresión (como i = i * 3 * 2)
        pattern_mult = rf'\b{var}\s*=\s*{var}\s*\*\s*(.+?)(?:;|\}})'
        match_mult = re.search(pattern_mult, content)
        if match_mult:
            expression = match_mult.group(1).strip()
            try:
                # Limpiar la expresión
                expression = expression.replace(' ', '').replace('}', '').strip()
                # Si es una expresión como "3*2", evaluarla
                if re.match(r'^[\d\*\+\-/\(\)]+$', expression):
                    multiplier = eval(expression)
                    step = f"*{multiplier}"
                else:
                    step = f"*{expression}"
            except:
                step = f"*{expression}"
            operations = 4
            return operations, step

        # Patrón para i = i / expresión
        pattern_div = rf'\b{var}\s*=\s*{var}\s*/\s*(.+?)(?:;|\}})'
        match_div = re.search(pattern_div, content)
        if match_div:
            expression = match_div.group(1).strip()
            try:
                expression = expression.replace(' ', '').replace('}', '').strip()
                if re.match(r'^[\d\*\+\-/\(\)]+$', expression):
                    divisor = eval(expression)
                    step = f"/{divisor}"
                else:
                    step = f"/{expression}"
            except:
                step = f"/{expression}"
            operations = 4
            return operations, step

        # Patrones más simples
        if re.search(rf'\b{var}\+\+|\+\+{var}\b', content):
            return 1, "+1"
        elif re.search(rf'\b{var}--|\--{var}\b', content):
            return 1, "-1"
        elif re.search(rf'\b{var}\s*\+=\s*(\d+)', content):
            match = re.search(rf'\b{var}\s*\+=\s*(\d+)', content)
            return 2, f"+{match.group(1)}" if match else "+1"
        elif re.search(rf'\b{var}\s*-=\s*(\d+)', content):
            match = re.search(rf'\b{var}\s*-=\s*(\d+)', content)
            return 2, f"-{match.group(1)}" if match else "-1"

        return operations, step

    def count_if_statements(self, block: str) -> int:
        """Cuenta las declaraciones IF en un bloque de código"""
        if not block.strip():
            return 0

        if_pattern = r'if\s*\(([^)]*(?:==|!=|<=|>=|<|>)[^)]*)\)'
        matches = re.findall(if_pattern, block, re.IGNORECASE)
        total_if_operations = 0

        if len(matches) > 0:
            for i, condition in enumerate(matches, 1):
                comparison_ops = 1
                arithmetic_ops = len(re.findall(r'[+\-*/]', condition))
                if_ops = comparison_ops + arithmetic_ops
                total_if_operations += if_ops

        return total_if_operations

    def parse_for_loop(self, for_statement: str) -> Optional[LoopInfo]:
        """Parsea una declaración for"""
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

    def parse_while_loop(self, while_statement: str, block_content: str) -> Optional[LoopInfo]:
        """Parsea una declaración while"""
        pattern = r'while\s*\(\s*(\w+)\s*([<>=!]+)\s*([^)]+)\)'

        match = re.search(pattern, while_statement, re.IGNORECASE)
        if match:
            var = match.group(1)
            operator = match.group(2).strip()
            end_condition = match.group(3).strip()

            # Analizar el contenido del bloque
            increment_ops, step = self.analyze_while_increment_operations(block_content, var)

            start = "0"
            end = end_condition

            loop = LoopInfo('while', var, start, end, step, self.current_level)
            loop.increment_expression = f"{var} modification in loop body"
            loop.complexity_type = self._determine_complexity_type(step)
            loop.condition = while_statement

            return loop
        return None

    def _extract_step_info(self, step_expr: str, var: str) -> str:
        """Extrae información del paso del bucle FOR"""
        step_expr = step_expr.replace(' ', '')

        # Casos simples: i++, ++i, i--, --i
        if f'{var}++' in step_expr or f'++{var}' in step_expr:
            return '+1'
        elif f'{var}--' in step_expr or f'--{var}' in step_expr:
            return '-1'

        # Casos con +=, -=
        elif f'{var}+=' in step_expr:
            pattern = rf'{var}\+=(\d+)'
            match = re.search(pattern, step_expr)
            if match:
                return f'+{match.group(1)}'
            return '+1'

        elif f'{var}-=' in step_expr:
            pattern = rf'{var}\-=(\d+)'
            match = re.search(pattern, step_expr)
            if match:
                return f'-{match.group(1)}'
            return '-1'

        # Casos con = (i = i + 2, i = i * 3, etc.)
        elif f'{var}=' in step_expr:
            # Patrón para i = i + número
            pattern_add = rf'{var}={var}\+(\d+)'
            match_add = re.search(pattern_add, step_expr)
            if match_add:
                return f'+{match_add.group(1)}'

            # Patrón para i = i - número
            pattern_sub = rf'{var}={var}\-(\d+)'
            match_sub = re.search(pattern_sub, step_expr)
            if match_sub:
                return f'-{match_sub.group(1)}'

            # Patrón para i = i * número
            pattern_mult = rf'{var}={var}\*(\d+)'
            match_mult = re.search(pattern_mult, step_expr)
            if match_mult:
                return f'*{match_mult.group(1)}'

            # Patrón para i = i / número
            pattern_div = rf'{var}={var}/(\d+)'
            match_div = re.search(pattern_div, step_expr)
            if match_div:
                return f'/{match_div.group(1)}'

        # Si no coincide con ningún patrón conocido
        return step_expr

    def _determine_complexity_type(self, step: str) -> str:
        """Determina el tipo de complejidad basado en el paso del bucle"""
        if step.startswith('+') or step.startswith('-'):
            return "linear"
        elif step.startswith('*') or step.startswith('/'):
            return "logarithmic"
        elif step in ['++', '--']:
            return "linear"
        else:
            return "constant"

    def calculate_loop_cycles(self, loop: LoopInfo) -> str:
        """Calcula el número de ciclos de un bucle"""
        start = loop.start
        end = loop.end
        step = loop.step

        if step == '+1':
            return f"({end} - {start} + 1)"
        elif step == '-1':
            return f"({start} - {end} + 1)"
        elif step.startswith('+'):
            increment = step[1:]
            return f"ceil(({end} - {start})/{increment})"
        elif step.startswith('-'):
            decrement = step[1:]
            return f"ceil(({start} - {end})/{decrement})"
        elif step.startswith('*'):
            multiplier = step[1:]
            return f"(log_{multiplier}({end}/{start}) + c)"
        elif step.startswith('/'):
            divisor = step[1:]
            return f"(log_{divisor}({start}/{end}) + c)"
        else:
            return f"f({start}, {end}, {step})"

    def count_operations_in_block(self, block: str) -> int:
        """Cuenta las operaciones en un bloque de código"""
        if not block.strip():
            return 0

        operations = 0

        # Contar IFs
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

    def _extract_while_block_content(self, lines: List[str], start_index: int) -> str:
        """Extrae el contenido completo de un bloque while"""
        content = []
        brace_count = 0
        found_opening_brace = False

        for i in range(start_index, len(lines)):
            line = lines[i].strip()

            if '{' in line:
                found_opening_brace = True
                brace_count += line.count('{')

            if found_opening_brace and i > start_index:
                content.append(line)

            if '}' in line:
                brace_count -= line.count('}')
                if brace_count <= 0:
                    break

        return ' '.join(content)

    def _identify_all_loops(self, code: str) -> List[LoopInfo]:
        """Identifica todos los bucles y sus niveles de anidamiento"""
        lines = code.split('\n')
        loops_info = []
        current_level = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('//'):
                i += 1
                continue

            if line.startswith('for'):
                loop = self.parse_for_loop(line)
                if loop:
                    loop.level = current_level
                    loops_info.append(loop)
                    current_level += 1

            elif line.startswith('while'):
                block_content = self._extract_while_block_content(lines, i)
                loop = self.parse_while_loop(line, block_content)
                if loop:
                    loop.level = current_level
                    loops_info.append(loop)
                    current_level += 1

            elif '}' in line and current_level > 0:
                current_level -= 1

            i += 1

        return loops_info

    def _extract_specific_loop_content(self, code: str, target_loop: LoopInfo) -> str:
        """Extrae solo las operaciones directas de un bucle específico"""
        lines = code.split('\n')
        content = []
        inside_target = False
        current_level = 0
        target_found = False
        skip_inner_loop = False
        inner_loop_level = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('//'):
                i += 1
                continue

            if (line.startswith('for') or line.startswith('while')) and not target_found:
                current_loop = None
                if line.startswith('for'):
                    current_loop = self.parse_for_loop(line)
                elif line.startswith('while'):
                    block_content = self._extract_while_block_content(lines, i)
                    current_loop = self.parse_while_loop(line, block_content)

                if (current_loop and current_loop.var == target_loop.var and
                        current_level == target_loop.level and
                        current_loop.loop_type == target_loop.loop_type):
                    inside_target = True
                    target_found = True
                current_level += 1

            elif (line.startswith('for') or line.startswith('while')) and inside_target:
                skip_inner_loop = True
                inner_loop_level = current_level
                current_level += 1

            elif '}' in line:
                current_level -= 1
                if skip_inner_loop and current_level <= inner_loop_level:
                    skip_inner_loop = False
                elif inside_target and current_level <= target_loop.level:
                    break

            elif inside_target and not skip_inner_loop:
                if (line.startswith('if') or
                        line.startswith('return') or
                        ('=' in line and not any(
                            op in line for op in ['==', '!=', '<=', '>=', '<', '>']) and
                         not line.startswith('for') and not line.startswith('while')) or
                        line.endswith(';')):
                    content.append(line)

            i += 1

        result = ' ; '.join(content)
        return result

    def analyze_nested_loops(self, code: str) -> Dict:
        """Método principal: Analiza bucles anidados y calcula la complejidad temporal"""
        loops_info = self._identify_all_loops(code)

        for loop in loops_info:
            loop_content = self._extract_specific_loop_content(code, loop)
            base_operations = self.count_operations_in_block(loop_content)

            if loop.loop_type == 'for':
                for_base = 1
                increment_ops = self.analyze_for_increment_operations(loop.increment_expression)
                total_ops = for_base + increment_ops
            else:  # while
                total_ops = 1

            loop.operations_count = base_operations + total_ops

        return self._calculate_time_complexity(loops_info)

    def _calculate_time_complexity(self, loops_info: List[LoopInfo]) -> Dict:
        """Calcula la complejidad temporal total"""
        if not loops_info:
            return {"complexity": "O(1)", "function": "T = 1", "steps": [], "analysis": "Sin bucles detectados"}

        loops_info.sort(key=lambda x: x.level, reverse=True)

        time_expressions = []
        analysis_steps = []  # Agregado de vuelta
        total_complexity = self._calculate_nested_complexity(loops_info)
        tiempo_acumulado = None

        for i, loop in enumerate(loops_info):
            cycles = self.calculate_loop_cycles(loop)
            operations = loop.operations_count

            if i == 0:
                time_expr = f"T_A = {cycles} * {operations} + 2"
                tiempo_acumulado = f"{cycles} * {operations} + 2"
            else:
                letter = chr(65 + i)
                time_expr = f"T_{letter} = {cycles} * ({operations} + ({tiempo_acumulado})) + 2"
                tiempo_acumulado = f"{cycles} * ({operations} + ({tiempo_acumulado})) + 2"

            time_expressions.append(time_expr)
            analysis_steps.append(f"Bucle {chr(65 + i)} ({loop.loop_type}): {loop.var} -> {time_expr}")

        return {
            "complexity": total_complexity,
            "function": time_expressions[-1] if time_expressions else "T = 1",
            "loops_count": len(loops_info),
            "steps": analysis_steps,  # Campo con información
            "analysis": self._generate_detailed_analysis(loops_info, time_expressions, total_complexity,
                                                         tiempo_acumulado)
        }

    def _calculate_nested_complexity(self, loops_info: List[LoopInfo]) -> str:
        """Calcula la complejidad de bucles anidados"""
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
        """Genera análisis detallado"""
        analysis = []
        analysis.append("=== ANÁLISIS DETALLADO ===")
        analysis.append("Fórmula: T = (numCiclos + c) × Tfor + 2")
        analysis.append("Nota: Operaciones del FOR = 1 (base) + operaciones del incremento")
        analysis.append("Nota: Operaciones del WHILE = 1 (evaluación condición) + operaciones del cuerpo")
        analysis.append("IMPORTANTE: El cálculo se hace de ADENTRO hacia AFUERA")

        time_values = []

        for i, loop in enumerate(loops_info):
            letter = chr(65 + i)

            analysis.append(f"\nBucle {letter}:")
            analysis.append(f"  Tipo: {loop.loop_type.upper()}")
            analysis.append(f"  Variable: {loop.var}")
            analysis.append(f"  Rango: {loop.start} hasta {loop.end}")
            analysis.append(f"  Paso: {loop.step}")
            analysis.append(f"  Tipo de complejidad: {loop.complexity_type}")

            if loop.loop_type == 'for':
                for_base = 1
                increment_ops = self.analyze_for_increment_operations(loop.increment_expression)
                total_for_ops = for_base + increment_ops
                base_ops = loop.operations_count - total_for_ops

                analysis.append(f"  Operaciones del código: {base_ops}")
                analysis.append(f"  Operaciones del FOR: +{total_for_ops}")
            else:  # while
                while_base = 1
                base_ops = loop.operations_count - while_base

                analysis.append(f"  Operaciones del código: {base_ops}")
                analysis.append(f"  Operaciones del WHILE: +{while_base}")
                if hasattr(loop, 'condition'):
                    analysis.append(f"  Condición: {loop.condition}")

            analysis.append(f"  Total operaciones propias: {loop.operations_count}")
            analysis.append(f"  Ciclos: {self.calculate_loop_cycles(loop)}")

            if i == 0:
                cycles = self.calculate_loop_cycles(loop)
                ops = loop.operations_count
                result = f"T_{letter} = {cycles} × {ops} + 2"
                analysis.append(f"  {result}")
                time_values.append(f"{cycles} × {ops} + 2")
            else:
                cycles = self.calculate_loop_cycles(loop)
                ops = loop.operations_count
                inner_value = time_values[i - 1]
                result = f"T_{letter} = {cycles} × ({ops} + ({inner_value})) + 2"
                analysis.append(f"  {result}")
                time_values.append(f"{cycles} × ({ops} + ({inner_value})) + 2")

        if len(time_values) > 0:
            final_letter = chr(65 + len(loops_info) - 1)
            analysis.append(f"\nFunción tiempo final: T_{final_letter} = {tiempo_final}")
            analysis.append(f"Complejidad asintótica: {final_complexity}")

        return "\n".join(analysis)


# Función de utilidad
def analyze_code_complexity(code: str) -> Dict:
    """Función de conveniencia para analizar código"""
    analyzer = TimeComplexityAnalyzer()
    return analyzer.analyze_nested_loops(code)