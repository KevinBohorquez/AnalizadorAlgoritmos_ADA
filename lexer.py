# lexer.py - Analizador Léxico para C++
import ply.lex as lex

# Tokens definidos
tokens = (
    'CLASS', 'PUBLIC', 'PRIVATE', 'PROTECTED', 'VOID', 'INT', 'FLOAT', 'DOUBLE',
    'CHAR', 'BOOL', 'RETURN', 'FOR', 'IF', 'ELSE', 'TRUE', 'FALSE', 'NULL',
    'CONST', 'STATIC', 'IDENTIFICADOR', 'NUMERO', 'CADENA', 'CARACTER',
    'SUMA', 'RESTA', 'MULTIPLICACION', 'DIVISION', 'MODULO', 'ASIGNACION',
    'INCREMENTO', 'DECREMENTO', 'SUMA_ASIG', 'RESTA_ASIG', 'MULT_ASIG', 'DIV_ASIG',
    'MENOR_QUE', 'MAYOR_QUE', 'MENOR_IGUAL', 'MAYOR_IGUAL', 'IGUAL_IGUAL', 'DISTINTO',
    'AND', 'OR', 'NOT', 'PUNTO', 'COMA', 'PUNTO_COMA', 'PARENTESIS_IZQ', 'PARENTESIS_DER',
    'LLAVE_IZQ', 'LLAVE_DER', 'CORCHETE_IZQ', 'CORCHETE_DER', 'SCOPE', 'PUNTERO', 'REFERENCIA',
    'COMENTARIO_LINEA', 'COMENTARIO_BLOQUE', 'NUEVA_LINEA', 'WHILE'
)

# Palabras reservadas
reserved = {
    'class': 'CLASS', 'public': 'PUBLIC', 'private': 'PRIVATE', 'protected': 'PROTECTED',
    'void': 'VOID', 'int': 'INT', 'float': 'FLOAT', 'double': 'DOUBLE', 'char': 'CHAR',
    'bool': 'BOOL', 'return': 'RETURN', 'for': 'FOR', 'if': 'IF', 'else': 'ELSE',
    'true': 'TRUE', 'false': 'FALSE', 'NULL': 'NULL', 'const': 'CONST', 'static': 'STATIC',
    'while': 'WHILE'
}

# Reglas de tokens (orden importante - más específicos primero)
t_SUMA_ASIG = r'\+='
t_RESTA_ASIG = r'-='
t_MULT_ASIG = r'\*='
t_DIV_ASIG = r'/='
t_INCREMENTO = r'\+\+'
t_DECREMENTO = r'--'
t_SUMA = r'\+'
t_RESTA = r'-'
t_MULTIPLICACION = r'\*'
t_DIVISION = r'/'
t_MODULO = r'%'
t_ASIGNACION = r'='
t_MENOR_IGUAL = r'<='
t_MAYOR_IGUAL = r'>='
t_IGUAL_IGUAL = r'=='
t_DISTINTO = r'!='
t_MENOR_QUE = r'<'
t_MAYOR_QUE = r'>'
t_AND = r'&&'
t_OR = r'\|\|'
t_NOT = r'!'
t_SCOPE = r'::'
t_PUNTERO = r'->'
t_REFERENCIA = r'&'
t_PUNTO = r'\.'
t_COMA = r','
t_PUNTO_COMA = r';'
t_PARENTESIS_IZQ = r'\('
t_PARENTESIS_DER = r'\)'
t_LLAVE_IZQ = r'\{'
t_LLAVE_DER = r'\}'
t_CORCHETE_IZQ = r'\['
t_CORCHETE_DER = r'\]'


def t_NUMERO(t):
    r'\d+(\.\d+)?([eE][+-]?\d+)?'
    """Reconoce números enteros y flotantes"""
    if '.' in t.value or 'e' in t.value.lower():
        t.value = float(t.value)
    else:
        t.value = int(t.value)
    return t


def t_IDENTIFICADOR(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    """Reconoce identificadores y palabras reservadas"""
    t.type = reserved.get(t.value, 'IDENTIFICADOR')
    return t


def t_CADENA(t):
    r'"([^"\\]|\\.)*"'
    """Reconoce cadenas de texto"""
    t.value = t.value[1:-1]  # Remover comillas
    return t


def t_CARACTER(t):
    r"'([^'\\]|\\.)*'"
    """Reconoce caracteres individuales"""
    t.value = t.value[1:-1]  # Remover comillas simples
    return t


def t_COMENTARIO_BLOQUE(t):
    r'/\*(.|\n)*?\*/'
    """Reconoce comentarios de bloque /* ... */"""
    t.lexer.lineno += t.value.count('\n')
    pass  # Los comentarios se ignoran


def t_COMENTARIO_LINEA(t):
    r'//.*'
    """Reconoce comentarios de línea //"""
    pass  # Los comentarios se ignoran


def t_NUEVA_LINEA(t):
    r'\n+'
    """Reconoce saltos de línea"""
    t.lexer.lineno += len(t.value)
    pass  # Los saltos de línea se ignoran


# Caracteres ignorados (espacios y tabulaciones)
t_ignore = ' \t'


def t_error(t):
    """Maneja caracteres no reconocidos"""
    print(f"Carácter ilegal '{t.value[0]}' en línea {t.lexer.lineno}")
    t.lexer.skip(1)


def create_lexer():
    """Crea y retorna una instancia del analizador léxico"""
    return lex.lex()


# Función de utilidad para probar el lexer
def test_lexer(code_string):
    """Prueba el lexer con una cadena de código"""
    lexer = create_lexer()
    lexer.input(code_string)

    tokens_found = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens_found.append((tok.type, tok.value, tok.lineno))

    return tokens_found